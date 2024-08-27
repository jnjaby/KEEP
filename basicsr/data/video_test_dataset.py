import glob
import torch
import os
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from collections import defaultdict
import numpy as np
import cv2

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_util import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceAligner


@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        global_meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.interval = opt['interval']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [],
                          'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'global_meta_info_file' in opt:
            with open(opt['global_meta_info_file'], 'r') as fin:
                subfolders = [line.split('/')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key)
                                 for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key)
                                 for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))[
                ::self.interval]
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))[
                ::self.interval]

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                logger.info(
                    f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        self.normalize = opt.get('normalize', False)

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(
            idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(
                0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        if self.normalize:
            normalize(imgs_lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        opt (dict): Same as VideoTestDataset. Unused opt:
        padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.need_align = opt.get('need_align', False)
        self.normalize = opt.get('normalize', False)

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
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            imgs_lq = read_img_seq(self.imgs_lq[folder])
            imgs_gt = read_img_seq(self.imgs_gt[folder])

        if self.need_align:
            clip_info_path = os.path.join(
                self.dataroot_meta_info, f'{folder}.txt')
            clip_info = []
            with open(clip_info_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line.startswith('0'):
                        clip_info.append(line)

            align_lqs, align_gts = [], []
            for frame_idx, (img_lq, img_gt) in enumerate(zip(imgs_lq, imgs_gt)):
                img_lq = tensor2img(img_lq) / 255.0
                img_gt = tensor2img(img_gt) / 255.0
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

        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        imgs_gt = torch.stack(img_gts, dim=0)
        imgs_lq = torch.stack(img_lqs, dim=0)

        if self.normalize:
            normalize(imgs_lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(imgs_gt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
