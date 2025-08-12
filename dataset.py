import config
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile

from utils import (
    pre_index,
    plot_image,
    normalize_bboxes
    
)

import cv2

import json
import os

class YOLODataset(Dataset):
    def __init__(self, 
                 split, 
                 S=[13, 26, 52],
                 transform=None
                   ):
        super().__init__()
        self.split = split
        self.S = S
        self.transform = transform
        self.id_annotations, self.id_images, self.id_categories = pre_index(
            os.path.join('datasets', config.DATASET, self.split, '_annotations.coco.json')
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        raw_path, W, H = self.id_images[index]
        raw_bboxes = self.id_annotations[index]

        bboxes = normalize_bboxes(raw_bboxes, W, H)
        image = np.array(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, 'train', raw_path)).convert("RGB"))

        class_labels = [box[0] for box in bboxes]
        coords_only = [box[1:] for box in bboxes]

        if self.transform:
            augmented = self.transform(image=image, bboxes=coords_only, class_labels=class_labels)
            image = augmented["image"]
            coords_only = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            bboxes = [[cls] + list(coord) for cls, coord in zip(class_labels, coords_only)]

        return super().__getitem__(index)
