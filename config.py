import os
import utils

import cv2

import albumentations as A

IMG_SIZE = 640
# Dataset Stuff

DATA_ROOT = 'datasets'
DATASET = 'COCO'

N_CLASSES = 20

train_transform = A.Compose([
    A.Mosaic(grid_yx=(2,2), target_size=[IMG_SIZE, IMG_SIZE]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.2, 0.2), rotate=(-30, 30), shear=(0.1, 0.1), border_mode=cv2.BORDER_CONSTANT) # FIX THIS BULLSHIT
    
    ],bbox_params=A.BboxParams(format='coco', label_fields=[])
    )


