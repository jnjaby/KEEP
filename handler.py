import os
import tempfile
import json
import base64
import subprocess
import cv2
import torch
import numpy as np
from PIL import Image
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from basicsr.archs.rrdbnet_arch import RRDBNet # Added for set_realesrgan
from basicsr.utils.realesrgan_utils import RealESRGANer # Added for set_realesrgan
from basicsr.utils.misc import gpu_is_available, get_device # get_device might be useful
from facelib.utils.face_restoration_helper import FaceRestoreHelper # Will be needed
from facelib.utils.misc import is_gray # for has_aligned processing in KEEP
from basicsr.utils.download_util import load_file_from_url # For loading models
from basicsr.utils.video_util import VideoReader, VideoWriter # For video I/O
from scipy.ndimage import gaussian_filter1d # For landmark smoothing
from tqdm import tqdm # For progress bars in processing loops

# Model paths (adjust if necessary for the serverless environment)
# Assuming 'checkpoints' directory is at the root of the deployment package
KEEP_MODEL_CHECKPOINT_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth"
KEEP_MODEL_DIR = "checkpoints/keep_models"
REALESRGAN_MODEL_URL = "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/RealESRGAN_x2plus.pth"
REALESRGAN_MODEL_DIR = "checkpoints/realesrgan_models"
# FACE_DETECTOR_MODEL_PATH = "checkpoints/other_models/retinaface_resnet50.pth" # This will be handled by FaceRestoreHelper

# Configuration defaults (can be overridden by job input)
DEFAULT_UPSCALE = 1
DEFAULT_BG_TILE = 400 # From inference_keep.py args
DEFAULT_FACE_UPSAMPLE = False # From inference_keep.py args
DEFAULT_BG_UPSAMPLER = 'realesrgan' # From inference_keep.py args, can be None
DEFAULT_DETECTION_MODEL = 'retinaface_resnet50' # From inference_keep.py args
DEFAULT_MAX_LENGTH = 20 # From inference_keep.py args, for video chunk processing
DEFAULT_MODEL_TYPE = 'KEEP' # From inference_keep.py args

# Global cache for models to avoid reloading on every call in the same worker instance
# This is a common optimization for serverless functions.
# Note: RunPod might have its own mechanisms or lifecycle for worker instances.
# This simple cache assumes the worker instance might process multiple jobs.
MODELS_CACHE = {}


# Placeholder for helper functions that will be integrated
# from inference_keep.py and hugging_face/app.py

def set_realesrgan(model_path=REALESRGAN_MODEL_URL, model_dir=REALESRGAN_MODEL_DIR, tile=DEFAULT_BG_TILE, device='cuda'):
    """Sets up the RealESRGAN model."""
    use_half = False
    if device == 'cuda' and torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660'] # GPUs that don't support f16
        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
            use_half = True
    
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2, # RealESRGAN_x2plus is a x2 model
    )
    
    # Download the model if a URL is provided and it's not already there
    if model_path.startswith('http'):
        actual_model_path = load_file_from_url(
            url=model_path,
            model_dir=model_dir,
            progress=True,
            file_name=None # Let it infer filename
        )
    else:
        actual_model_path = model_path

    upsampler = RealESRGANer(
        scale=2,
        model_path=actual_model_path,
        model=model,
        tile=tile,
        tile_pad=40, # Default from inference_keep.py
        pre_pad=0,   # Default from inference_keep.py
        half=use_half,
        device=device
    )

    if device == 'cpu':
        import warnings
        warnings.warn('Running RealESRGAN on CPU is slow. Consider using a GPU.', category=RuntimeWarning)
    return upsampler

# Model configurations (adapted from inference_keep.py)
MODEL_CONFIGS = {
    'KEEP': {
        'architecture': {
            'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
            'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
            'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4,
            'cfa_dim': 256, 'cond': 1,
        },
        'checkpoint_url': KEEP_MODEL_CHECKPOINT_URL,
        'checkpoint_dir': KEEP_MODEL_DIR
    },
    # Add 'Asian' model config if needed, similar to inference_keep.py
}

def load_keep_model(model_type=DEFAULT_MODEL_TYPE, device='cuda'):
    """Loads the KEEP restoration model."""
    if model_type in MODELS_CACHE and 'keep_model' in MODELS_CACHE[model_type]:
        print(f"Using cached KEEP model: {model_type}")
        return MODELS_CACHE[model_type]['keep_model']

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_type]
    
    # Ensure the KEEP architecture is registered if not using basicsr standard way
    # Assuming ARCH_REGISTRY.get('KEEP') works as expected from basicsr installation
    net = ARCH_REGISTRY.get('KEEP')(**config['architecture']).to(device)

    ckpt_path = load_file_from_url(
        url=config['checkpoint_url'],
        model_dir=config['checkpoint_dir'],
        progress=True,
        file_name=None) # Let it infer filename
    
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=True) # Ensure loading on correct device
    net.load_state_dict(checkpoint['params_ema'])
    net.eval()

    if model_type not in MODELS_CACHE:
        MODELS_CACHE[model_type] = {}
    MODELS_CACHE[model_type]['keep_model'] = net
    print(f"KEEP model loaded and cached: {model_type}")
    return net

# Helper function from inference_keep.py for landmark smoothing
def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)

    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(
            x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence


# video_to_frames and frames_to_video are effectively handled by VideoReader and VideoWriter
# so direct helper functions for those might not be strictly necessary if we adapt the main loop.

def align_face(frame, landmarks, face_helper):
    """Aligns a single face in a frame using FaceRestoreHelper methods."""
    # This is a conceptual placeholder. Actual alignment is part of face_helper.read_image()
    # and subsequent calls to align_warp_face() after landmarks are set.
    # For KEEP, the process is:
    # 1. face_helper.read_image(img)
    # 2. face_helper.all_landmarks_5 = [landmarks_for_this_frame]
    # 3. face_helper.align_warp_face()
    # The cropped face is then in face_helper.cropped_faces[0]
    pass


def process_video_with_keep(
    input_video_path, 
    output_video_path, 
    keep_model, 
    face_helper, 
    bg_upsampler, 
    face_upsampler,
    upscale_factor,
    detection_model_name, # For re-confirming or if needed
    has_aligned_input=False, # If input video is already aligned faces
    only_center_face=True,
    max_length=DEFAULT_MAX_LENGTH,
    save_video_fps=25, # Default FPS, consider getting from input video
    device='cuda'
    ):
    """
    Processes a video using the KEEP model and other enhancements.
    Adapted from the main processing loop in inference_keep.py.
    """
    
    # 1. Read video frames
    print(f"Reading video: {input_video_path}")
    vidreader = VideoReader(input_video_path)
    input_img_list = []
    while True:
        img = vidreader.get_frame()
        if img is None:
            break
        input_img_list.append(img)
    original_fps = vidreader.get_fps()
    vidreader.close()

    if not input_img_list:
        raise ValueError("No frames read from video. Check video file or path.")
    
    print(f"Video read successfully. Number of frames: {len(input_img_list)}, FPS: {original_fps}")
    
    # Use original_fps if save_video_fps is not specified or invalid
    if save_video_fps <= 0:
        save_video_fps = original_fps


    # 2. Face detection and landmark smoothing (if not has_aligned_input)
    if not has_aligned_input:
        print('Detecting keypoints and smoothing alignment...')
        raw_landmarks_list = []
        for i, img in enumerate(tqdm(input_img_list, desc="Detecting landmarks")):
            face_helper.clean_all() # Important for each frame
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True
            )
            if num_det_faces == 1:
                raw_landmarks_list.append(face_helper.all_landmarks_5[0].reshape((10,)))
            else: # No face or multiple faces (and only_center_face handled it)
                raw_landmarks_list.append(np.array([np.nan] * 10))
        
        raw_landmarks_np = np.array(raw_landmarks_list)
        for i in range(10): # Smooth each of the 5 (x,y) coordinates
            raw_landmarks_np[:, i] = interpolate_sequence(raw_landmarks_np[:, i])
        
        # Apply Gaussian smoothing
        smoothed_landmarks = gaussian_filter1d(raw_landmarks_np, sigma=5, axis=0).reshape(len(input_img_list), 5, 2)
        print("Landmark detection and smoothing complete.")

    # 3. Prepare cropped faces for KEEP model
    cropped_face_tensors = []
    print("Aligning and preparing face tensors...")
    for i, img in enumerate(tqdm(input_img_list, desc="Aligning faces")):
        if not has_aligned_input:
            face_helper.clean_all()
            face_helper.read_image(img) # Read original image for context
            face_helper.all_landmarks_5 = [smoothed_landmarks[i]] # Set smoothed landmarks
            face_helper.align_warp_face() # Align and crop
        else: # If input is already aligned (e.g., a video of cropped faces)
            # This path might need adjustment based on how 'has_aligned' is truly used.
            # Typically, KEEP expects 512x512 aligned faces.
            img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img_resized, threshold=10) # Check if grayscale
            face_helper.cropped_faces = [img_resized] # Directly use the image

        if not face_helper.cropped_faces:
            # Handle cases where alignment might fail or no face is present
            # For now, creating a dummy black tensor. Robust handling needed.
            print(f"Warning: No cropped face for frame {i}. Using a black image.")
            dummy_face = np.zeros((512, 512, 3), dtype=np.uint8)
            cropped_face_tensor = img2tensor(dummy_face / 255., bgr2rgb=True, float32=True)
        else:
            cropped_face_tensor = img2tensor(face_helper.cropped_faces[0] / 255., bgr2rgb=True, float32=True)
        
        normalize(cropped_face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_tensors.append(cropped_face_tensor)
    
    cropped_faces_torch = torch.stack(cropped_face_tensors, dim=0).unsqueeze(0).to(device) # (1, num_frames, C, H, W)
    print("Face tensors prepared.")

    # 4. Face restoration with KEEP model (in chunks)
    print('Restoring faces with KEEP model...')
    restored_faces_list = []
    video_length = cropped_faces_torch.shape[1]
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, video_length, max_length), desc="KEEP Processing Chunks"):
            end_idx = min(start_idx + max_length, video_length)
            # Handle single frame chunks if necessary (KEEP model might require at least 2 for some ops)
            if end_idx - start_idx == 1:
                 # Duplicate the single frame to make a sequence of 2
                chunk_output = keep_model(cropped_faces_torch[:, start_idx:end_idx, ...].repeat(1,2,1,1,1), need_upscale=False)[:, 0:1, ...]
            else:
                chunk_output = keep_model(cropped_faces_torch[:, start_idx:end_idx, ...], need_upscale=False)
            restored_faces_list.append(chunk_output)
    
    output_tensor = torch.cat(restored_faces_list, dim=1).squeeze(0) # (num_frames, C, H, W)
    del cropped_faces_torch, restored_faces_list # Free memory
    torch.cuda.empty_cache()

    # Convert restored face tensors to images (list of BGR numpy arrays)
    final_restored_faces = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in output_tensor]
    print("Face restoration complete.")

    # 5. Paste faces back and create final video frames
    print('Pasting faces back and creating final frames...')
    final_frames = []
    vidwriter = VideoWriter(output_video_path, input_img_list[0].shape[0], input_img_list[0].shape[1], save_video_fps)

    for i, original_frame in enumerate(tqdm(input_img_list, desc="Pasting faces")):
        face_helper.clean_all()
        
        if has_aligned_input: # If input was aligned, output is just the restored face
            # (Potentially resized or with background if specified, but simpler case here)
            # This might need upsampling if upscale_factor > 1 and face_upsampler is present
            output_frame = final_restored_faces[i]
            if upscale_factor > 1 and face_upsampler:
                 output_frame = face_upsampler.enhance(output_frame, outscale=upscale_factor)[0]
        else:
            face_helper.read_image(original_frame) # Read original frame for pasting
            face_helper.all_landmarks_5 = [smoothed_landmarks[i]] # Set landmarks for inverse affine
            face_helper.align_warp_face() # This is needed to set internal state for pasting

            face_helper.add_restored_face(final_restored_faces[i].astype('uint8'))

            bg_img_upsampled = None
            if bg_upsampler is not None:
                bg_img_upsampled = bg_upsampler.enhance(original_frame, outscale=upscale_factor)[0]
            
            face_helper.get_inverse_affine(None) # Prepare for pasting

            output_frame = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img_upsampled, # Background, possibly upscaled
                draw_box=False, # No bounding box in final output
                face_upsampler=face_upsampler if upscale_factor > 1 else None # Upsample face if specified
            )
        
        vidwriter.write_frame(output_frame)
    
    vidwriter.close()
    print(f"Processed video saved to {output_video_path}")
    return output_video_path


def handler(job):
    """
    Handles the serverless request for video processing.
    """
    try:
        payload = job["input"]
        video_url = payload.get("video_url")
        # Get optional parameters from payload or use defaults
        upscale = payload.get("upscale", DEFAULT_UPSCALE)
        bg_upsampler_name = payload.get("bg_upsampler", DEFAULT_BG_UPSAMPLER) # e.g., "realesrgan" or None
        face_upsample = payload.get("face_upsample", DEFAULT_FACE_UPSAMPLE)
        detection_model_name = payload.get("detection_model", DEFAULT_DETECTION_MODEL)
        bg_tile = payload.get("bg_tile", DEFAULT_BG_TILE)
        model_type = payload.get("model_type", DEFAULT_MODEL_TYPE) # e.g. "KEEP"

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        if not video_url:
            return {"error": "Missing video_url in payload"}

        # 1. Download the video
        # Using a temporary file for the downloaded video
        # Ensure the temp directory used by NamedTemporaryFile is writable in RunPod
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_file:
            input_video_path = tmp_video_file.name
        
        print(f"Downloading video from {video_url} to {input_video_path}")
        subprocess.run(["wget", "-O", input_video_path, video_url], check=True)
        print("Video downloaded.")

        # 2. Load models
        print("Loading models...")
        face_helper_instance = None # Ensure it's defined for cleanup
        try:
            keep_model_instance = load_keep_model(model_type=model_type, device=device)
            
            bg_upsampler_instance = None
            if bg_upsampler_name == 'realesrgan':
                bg_upsampler_instance = set_realesrgan(tile=bg_tile, device=device) 
            
            face_upsampler_instance = None
            if face_upsample:
                if bg_upsampler_instance is not None: 
                    face_upsampler_instance = bg_upsampler_instance
                else: 
                    face_upsampler_instance = set_realesrgan(tile=bg_tile, device=device)

            face_helper_instance = FaceRestoreHelper(
                upscale, 
                face_size=512, 
                crop_ratio=(1, 1), 
                det_model=detection_model_name,
                save_ext='png', 
                use_parse=True, 
                device=device
            )
            print("Models loaded.")

        except Exception as e:
            print(f"Error loading models: {e}")
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
            return {"error": f"Model loading failed: {str(e)}"}


        # 3. Process the video
        processed_video_final_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output_file:
                temp_output_video_path = tmp_output_file.name
            
            print(f"Starting video processing. Output will be at {temp_output_video_path}")
            processed_video_final_path = process_video_with_keep(
                input_video_path=input_video_path,
                output_video_path=temp_output_video_path,
                keep_model=keep_model_instance,
                face_helper=face_helper_instance,
                bg_upsampler=bg_upsampler_instance,
                face_upsampler=face_upsampler_instance,
                upscale_factor=upscale,
                detection_model_name=detection_model_name, # Pass along for consistency
                # has_aligned_input can be a job parameter if needed
                device=device,
                save_video_fps=payload.get("save_video_fps", 0) # Get from payload or use original
            )
            print(f"Video processing finished. Output at {processed_video_final_path}")

        except Exception as e:
            print(f"Error during video processing: {e}")
            # Clean up temp files if processing fails
            if os.path.exists(input_video_path): os.remove(input_video_path)
            if temp_output_video_path and os.path.exists(temp_output_video_path): os.remove(temp_output_video_path)
            return {"error": f"Video processing failed: {str(e)}"}


        # 4. Prepare the response
        # Option 1: Return the path to the processed video (if accessible by the caller)
        # For serverless, returning the video data itself (base64) is an option,
        # but for large video files, it's better to upload to S3 (or similar)
        # and return a URL. For RunPod, returning the path might be okay if the
        # file system is ephemeral but accessible for a short period, or if RunPod
        # handles artifact storage.
        
        # For now, let's assume the path is sufficient or will be handled by RunPod.
        # If base64 is needed:
        # with open(processed_video_final_path, "rb") as video_file:
        #     video_data_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        # response = {"processed_video_base64": video_data_base64}

        response = {
            "processed_video_path": processed_video_final_path,
            "message": "Video processing completed successfully."
        }

    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        return {"error": f"Failed to download video: {e.strerror} (URL: {e.cmd[-1]})"} # Provide more context
    except FileNotFoundError as e:
        print(f"Error: File not found (e.g., model file or temp file): {e}")
        return {"error": f"Processing error: File not found - {e.filename}"}
    except Exception as e:
        # Log the full traceback for unexpected errors
        import traceback
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred during processing: {str(e)}"}
    finally:
        # Clean up temporary files
        if 'input_video_path' in locals() and input_video_path and os.path.exists(input_video_path):
            print(f"Cleaning up input video: {input_video_path}")
            os.remove(input_video_path)
        
        # The processed_video_final_path should NOT be deleted here if it's the intended output.
        # RunPod (or the calling service) will handle its lifecycle.
        # If it was a temporary path that was then copied to a final destination (e.g., S3),
        # then the temporary one could be deleted.
        # For now, we assume processed_video_final_path is the path RunPod expects.

        # Clean up any models from FaceRestoreHelper if they have a cleanup method
        if face_helper_instance and hasattr(face_helper_instance, 'clean_all'):
             # This primarily cleans intermediate data for the next frame,
             # but good to call. Actual model unloading is not standard.
             face_helper_instance.clean_all()
        
        # Clear torch CUDA cache if a GPU was used
        if device == 'cuda':
            torch.cuda.empty_cache()
            print("Cleared PyTorch CUDA cache.")

    return response

if __name__ == "__main__":
    # Example usage (for local testing)
    # This part will be executed by RunPod in the serverless environment.
    # For local testing, you might need to mock the 'job' object
    # and ensure model paths are correct for your local setup.
    # Also, ensure 'checkpoints' directory exists or models are downloaded.

    # Create a dummy job for testing
    # You'd need a publicly accessible video URL here
    # Ensure the checkpoints directory exists and models can be downloaded to it for local test
    if not os.path.exists(REALESRGAN_MODEL_DIR):
        os.makedirs(REALESRGAN_MODEL_DIR, exist_ok=True)
    if not os.path.exists(KEEP_MODEL_DIR):
        os.makedirs(KEEP_MODEL_DIR, exist_ok=True)
        
    sample_job_for_local_test = {
        "input": {
            "video_url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4", # Replace with a valid test URL
            # Add other optional parameters for testing if needed
            # "upscale": 1,
            # "bg_upsampler": "realesrgan",
            # "face_upsample": False,
        }
    }
    
    # Simulate model loading for local test
    print("Simulating model loading for local test (if not already loaded by handler)...")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # try:
    #     bg_upsampler = set_realesrgan(device=device)
    #     print("RealESRGAN upsampler loaded for testing.")
    # except Exception as e:
    #     print(f"Could not load RealESRGAN for local test: {e}")

    print("Starting handler for local test...")
    result = handler(sample_job_for_local_test)
    print("\nHandler Result (Local Test):")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # print(f"Processed video (base64): {result.get('processed_video_base64', '')[:100]}...")
        print(f"Processed video path: {result.get('processed_video_path')}")
        print(f"Message: {result.get('message')}")
        output_path = result.get('processed_video_path')
        if output_path and os.path.exists(output_path):
             print(f"Output file '{output_path}' exists. Size: {os.path.getsize(output_path)} bytes.")
             # os.remove(output_path) # Clean up the dummy output for local test
        elif output_path:
            print(f"Output file '{output_path}' not found (this is expected if only simulation ran).")


    print("Local test finished.")

# Further steps:
# 1. Integrate KEEP model loading and core restoration logic.
# 2. Implement video_to_frames and frames_to_video.
# 3. Integrate FaceRestoreHelper and face detection/alignment.
# 4. Wire up the main processing pipeline in `process_video_with_keep`.
# 5. Refine parameter passing from job input to various components.
# 6. Enhance error handling, logging, and cleanup.
# 7. Test thoroughly.
