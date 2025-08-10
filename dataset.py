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

import json
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

        image = np.array(Image.open(os.path.join(config.DATA_ROOT, config.DATASET, 'train', raw_path)).convert("RGB"))

        return super().__getitem__(index)
