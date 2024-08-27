import os
import random
from pathlib import Path

from PIL import Image
import cv2
import ffmpeg
import io
import av
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.data.degradations import (random_add_gaussian_noise,
                                       random_mixed_kernels)
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, img2tensor, imfrombytes, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from facelib.utils.face_restoration_helper import FaceAligner
from torch.utils import data as data


@DATASET_REGISTRY.register()
class VFHQRealDegradationDataset(data.Dataset):
    """Support for blind setting adopted in paper. We excludes the random scale compared to GFPGAN.

    This dataset is adopted in BasicVSR.

    The degradation order is blur+downsample+noise

    Directly read image by cv2. Generate LR images online.
    NOTE: The specific degradation order is blur-noise-downsample-crf-upsample

    The keys are generated from a meta info txt file.

    Key format: subfolder-name/clip-length/frame-name
    Key examples: "id00020#t0bbIRgKKzM#00381.txt#000.mp4/00000152/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_clip_meta_info (srt): Data root path for meta info of each gt clip.
            global_meta_info_file (str): Path for global meta information file.
            io_backend (dict): IO backend type and other kwarg.
            num_frame (int): Window size for input frames.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
    """

    def __init__(self, opt):
        super(VFHQRealDegradationDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])

        self.num_frame = opt['num_frame']
        self.scale = opt['scale']
        self.need_align = opt.get('need_align', False)
        self.normalize = opt.get('normalize', False)

        self.keys = []
        with open(opt['global_meta_info_file'], 'r') as fin:
            for line in fin:
                real_clip_path = '/'.join(line.split('/')[:-1])
                clip_length = line.split('/')[-1]
                clip_length = int(clip_length)
                self.keys.extend(
                    [f'{real_clip_path}/{clip_length:08d}/{frame_idx:08d}' for frame_idx in range(int(clip_length))])
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # degradations
        # blur
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_x_sigma = opt['blur_x_sigma']
        self.blur_y_sigma = opt['blur_y_sigma']
        # noise
        self.noise_range = opt['noise_range']
        # resize
        self.resize_prob = opt['resize_prob']
        # crf
        self.crf_range = opt['crf_range']
        # codec
        self.vcodec = opt['vcodec']
        self.vcodec_prob = opt['vcodec_prob']

        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'x_sigma: [{", ".join(map(str, self.blur_x_sigma))}], '
                    f'y_sigma: [{", ".join(map(str, self.blur_y_sigma))}], ')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(
            f'CRF compression: [{", ".join(map(str, self.crf_range))}]')
        logger.info(f'Codec: [{", ".join(map(str, self.vcodec))}]')

        if self.need_align:
            self.dataroot_meta_info = opt['dataroot_meta_info']
            self.face_aligner = FaceAligner(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        real_clip_path = '/'.join(key.split('/')[:-2])
        clip_length = int(key.split('/')[-2])
        frame_idx = int(key.split('/')[-1])
        clip_name = real_clip_path.split('/')[-1]

        paths = sorted(list(scandir(os.path.join(
            self.gt_root, clip_name))))

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # exceed the length, re-select a new clip
        while (clip_length - self.num_frame * interval) < 0:
            interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        # print(self.num_frame, type(self.num_frame))
        # print(interval, type(interval))
        start_frame_idx = frame_idx - self.num_frame // 2 * interval
        end_frame_idx = frame_idx + self.num_frame // 2 * interval

        # flag = (start_frame_idx < 0) or (end_frame_idx > clip_length)
        # print(key, start_frame_idx, end_frame_idx, interval, flag)
        # each clip has 100+ frames
        while (start_frame_idx < 0) or (end_frame_idx > clip_length):
            frame_idx = random.randint(self.num_frame//2 * interval,
                                       clip_length - self.num_frame//2 * interval)
            start_frame_idx = frame_idx - self.num_frame // 2 * interval
            end_frame_idx = frame_idx + self.num_frame // 2 * interval
        neighbor_list = list(
            range(start_frame_idx, end_frame_idx, interval))
        # print(start_frame_idx, end_frame_idx, frame_idx, interval)
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (
            f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the neighboring GT frames
        img_gts = []

        if self.need_align:
            clip_info_path = os.path.join(
                self.dataroot_meta_info, f'{clip_name}.txt')
            clip_info = []
            with open(clip_info_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line.startswith('0'):
                        clip_info.append(line)

        for neighbor in neighbor_list:
            assert paths[neighbor] == clip_info[neighbor].split(' ')[0], \
                f'{clip_name}: Mismatch frame {paths[neighbor]} and {clip_info[neighbor]}'
            # img_gt_path = os.path.join(
            #     self.gt_root, clip_name, f'{neighbor:08d}.png')
            img_gt_path = os.path.join(
                self.gt_root, clip_name, paths[neighbor])
            # img_bytes = self.file_client.get(img_gt_path, 'gt')
            # img_gt = imfrombytes(img_bytes, float32=True)
            # img_gt = cv2.imread(img_gt_path) / 255.0
            img_gt = np.asarray(Image.open(img_gt_path))[:, :, ::-1] / 255.0
            img_gts.append(img_gt)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        # ------------- generate LQ frames --------------#
        # add blur
        kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, self.blur_kernel_size, self.blur_x_sigma,
                                      self.blur_y_sigma)
        img_lqs = [cv2.filter2D(v, -1, kernel) for v in img_gts]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]
        # downsample
        original_height, original_width = img_gts[0].shape[0:2]
        resize_type = random.choices(
            [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC], self.resize_prob)[0]
        resized_height, resized_width = int(
            original_height // self.scale), int(original_width // self.scale)
        # ensure the resized_height and resized_width are even numbers
        img_lqs = [cv2.resize(v, (resized_width, resized_height),
                              interpolation=resize_type) for v in img_lqs]
        # add noise
        img_lqs = [
            random_add_gaussian_noise(v, self.noise_range, gray_prob=0.5, clip=True, rounds=False) for v in img_lqs
        ]

        # ffmpeg
        crf = np.random.randint(self.crf_range[0], self.crf_range[1])
        codec = random.choices(self.vcodec, self.vcodec_prob)[0]

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = resized_height
            stream.width = resized_width
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': str(crf)}

            for img_lq in img_lqs:
                img_lq = np.clip(img_lq * 255, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img_lq, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        img_lqs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    img_lqs.append(frame.to_rgb().to_ndarray() / 255.)

        assert len(img_lqs) == len(img_gts), 'Wrong length'
        # ------------ Align -------------#
        if self.need_align:
            align_lqs, align_gts = [], []
            for frame_idx, (img_lq, img_gt) in enumerate(zip(img_lqs, img_gts)):
                landmarks_str = clip_info[frame_idx].split(' ')[1:]
                # print(clip_name, paths[neighbor], landmarks_str)
                landmarks = np.array([float(x)
                                     for x in landmarks_str]).reshape(5, 2)
                self.face_aligner.clean_all()
                # align and warp each face
                img_lq, img_gt = self.face_aligner.align_pair_face(
                    img_lq, img_gt, landmarks)
                align_lqs.append(img_lq)
                align_gts.append(img_gt)
            img_lqs, img_gts = align_lqs, align_gts

        # ------------- end --------------#
        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        if self.normalize:
            normalize(img_lqs, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gts, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class SingleVFHQDataset(data.Dataset):
    """Support for blind setting adopted in paper. We excludes the random scale compared to GFPGAN.

    This dataset is adopted in BasicVSR.

    The degradation order is blur+downsample+noise

    Note that we skip the low quality frames within the VFHQ clip.
    Directly read image by cv2. Generate LR images online.
    NOTE: The specific degradation order is blur-noise-downsample-crf-upsample

    The keys are generated from a meta info txt file.

    Key format: subfolder-name/clip-length/frame-name
    Key examples: "id00020#t0bbIRgKKzM#00381.txt#000.mp4/00000152/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_clip_meta_info (srt): Data root path for meta info of each gt clip.
            global_meta_info_file (str): Path for global meta information file.
            io_backend (dict): IO backend type and other kwarg.
            num_frame (int): Window size for input frames.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
    """

    def __init__(self, opt):
        super(SingleVFHQDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.normalize = opt.get('normalize', False)
        self.need_align = opt.get('need_align', False)
        logger = get_root_logger()

        self.keys = []
        with open(opt['global_meta_info_file'], 'r') as fin:
            for line in fin:
                real_clip_path = '/'.join(line.split('/')[:-1])
                clip_length = line.split('/')[-1]
                clip_length = int(clip_length)
                self.keys.extend(
                    [f'{real_clip_path}/{clip_length:08d}/{frame_idx:08d}' for frame_idx in range(int(clip_length))])
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        if self.need_align:
            self.dataroot_meta_info = opt['dataroot_meta_info']
            self.face_aligner = FaceAligner(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        real_clip_path = '/'.join(key.split('/')[:-2])
        clip_length = int(key.split('/')[-2])
        frame_idx = int(key.split('/')[-1])

        # get the neighboring GT frames
        flag = real_clip_path.split('/')[0]
        clip_name = real_clip_path.split('/')[-1]

        paths = sorted(list(scandir(os.path.join(
            self.gt_root, clip_name))))

        assert len(paths) == clip_length, "Wrong length of frame list"

        img_gt_path = os.path.join(
            self.gt_root, clip_name, paths[frame_idx])
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # alignment
        if self.need_align:
            clip_info_path = os.path.join(
                self.dataroot_meta_info, f'{clip_name}.txt')
            clip_info = []
            with open(clip_info_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line.startswith('0'):
                        clip_info.append(line)

            landmarks_str = clip_info[frame_idx].split(' ')[1:]
            landmarks = np.array([float(x)
                                  for x in landmarks_str]).reshape(5, 2)
            self.face_aligner.clean_all()
            # align and warp each face
            img_gt = self.face_aligner.align_single_face(img_gt, landmarks)

        # augmentation - flip, rotate
        img_gt = augment(img_gt, self.opt['use_flip'], self.opt['use_rot'])
        img_in = img_gt

        # ------------- end --------------#
        img_in, img_gt = img2tensor([img_in, img_gt])
        if self.normalize:
            normalize(img_in, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'in': img_in, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

