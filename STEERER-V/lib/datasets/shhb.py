
import os

import cv2
import numpy as np
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
from .nwpu import NWPU

def modify_path(path):
    # Split the path into parts
    parts = path.split(os.sep)

    # Remove the 'img1' part
    if 'img1' in parts:
        parts.remove('img1')

    # Extract the filename and directory
    *dirs, filename = parts

    # Remove leading zeros from the filename (excluding the extension)
    name, ext = os.path.splitext(filename)
    name = name.lstrip('0')

    # Reassemble the path
    new_path = os.sep.join(dirs + [f"{name}"])

    return new_path

class SHHB(NWPU):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5, 1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 scale=1,
                 testing=False,
                 phase='train', temporal=False):
        self.testing=testing
        

        self.phase=phase
        super(SHHB, self).__init__(
            root,
            list_path,
            num_samples,
            num_classes,
            multi_scale,
            flip,
            ignore_label,
            base_size,
            crop_size,
            min_unit ,
            center_crop_test,
            downsample_rate,
            scale_factor,
            mean,
            std, scale=1, testing=self.testing, phase=self.phase)





    def read_files(self):
        box_gt_Info = self.read_box_gt(os.path.join(self.root, 'val_gt_loc.txt'))
        files = []

        for item in self.img_list:
            # import pdb
            # pdb.set_trace()
            image_id = item[0]
            if 'val' in self.list_path:
                self.box_gt.append(box_gt_Info[int(image_id)])
            files.append({
                "img": 'images/' + image_id,
                "label": 'jsons/' + modify_path(image_id) + '.json',
                "name": image_id,
                "weight": 1
            })
        return files