# Base image from RunPod with PyTorch, CUDA, and Python
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies like ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Ensure runpod, ffmpeg-python, basicsr, facelib (via setup.py), scipy, tqdm are covered
# basicsr and facelib will be installed via setup.py develop later
# Add any other specific dependencies from handler.py if not covered by original requirements
RUN pip install --no-cache-dir -r requirements.txt \
    runpod \
    ffmpeg-python \
    scipy \
    tqdm 
    # basicsr and facelib are handled by setup.py develop

# Copy application code and necessary directories
COPY handler.py .
COPY setup.py .
COPY basicsr ./basicsr
COPY facelib ./facelib
COPY options ./options
# The 'weights' directory from the repo is mostly a placeholder or for local use.
# We will download models directly into a 'checkpoints' directory as expected by handler.py.

# Create directories for model checkpoints as expected by handler.py and facelib
RUN mkdir -p checkpoints/keep_models && \
    mkdir -p checkpoints/realesrgan_models && \
    mkdir -p checkpoints/facelib_models && \
    mkdir -p checkpoints/other_models && \
    mkdir -p weights/facelib # For models downloaded by facelib itself if not placed in checkpoints

# Download model checkpoints
# KEEP Models
RUN wget --tries=3 -O checkpoints/keep_models/KEEP-b76feb75.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth && \
    wget --tries=3 -O checkpoints/keep_models/KEEP_Asian-4765ebe0.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP_Asian-4765ebe0.pth

# RealESRGAN Model for background and general upsampling
RUN wget --tries=3 -O checkpoints/realesrgan_models/RealESRGAN_x2plus.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/RealESRGAN_x2plus.pth

# Face Detection Models (RetinaFace - these are used by FaceRestoreHelper)
# Updated to use jnjaby/KEEP URLs and download to checkpoints/facelib_models/
RUN wget --tries=3 -O checkpoints/facelib_models/detection_Resnet50_Final.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_Resnet50_Final.pth && \
    wget --tries=3 -O checkpoints/facelib_models/detection_mobilenet0.25_Final.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/detection_mobilenet0.25_Final.pth

# YOLOv5 Face Detection Models (alternative detectors in FaceRestoreHelper)
# Ensure these paths match what FaceRestoreHelper/YOLOv5Face expects if used,
# or adjust FaceRestoreHelper to look for them in checkpoints/other_models.
# For now, downloading to a common place.
RUN wget --tries=3 -O checkpoints/other_models/yolov5n-face.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5n-face.pth && \
    wget --tries=3 -O checkpoints/other_models/yolov5l-face.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/yolov5l-face.pth

# Face Parsing Model (used by FaceRestoreHelper)
# Updated to use jnjaby/KEEP URL and download to checkpoints/facelib_models/
RUN wget --tries=3 -O checkpoints/facelib_models/parsing_parsenet.pth https://github.com/jnjaby/KEEP/releases/download/v1.0.0/parsing_parsenet.pth

# GMFlow Model has been removed as it's not directly used by the handler.

# Install the project (basicsr, facelib) in editable mode
# This makes sure that the custom versions of basicsr and facelib are used.
RUN python setup.py develop

# Set environment variables if necessary (e.g., Python path)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# RunPod serverless workers typically don't need an explicit ENTRYPOINT or CMD in the Dockerfile,
# as the platform handles invoking the handler function.
# However, ensuring the environment is correctly set up is key.

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y && \
    apt-get clean

# The handler.py is expected to be triggered by the RunPod environment.
# Default command can be python -m runpod.serverless.rp_fastapi
# CMD ["python", "-m", "runpod.serverless.rp_fastapi"]
# However, usually not needed to specify for RunPod.
