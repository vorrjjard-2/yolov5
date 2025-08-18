import os
import utils

import torch

import cv2

import albumentations as A

IMG_SIZE = 640
# Dataset Stuff

DATA_ROOT = 'datasets'
DATASET = 'COCO'

N_CLASSES = 20


#Training Config

BATCH_SIZE = 8

# Augmentations

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    #A.Perspective(scale=(0.0, 0.0005), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=(114,114,114), p=0.2),
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(0.2, 0.2), 
        rotate=(-30, 30), 
        shear=(0.1, 0.1), 
        border_mode=cv2.BORDER_CONSTANT), # FIX THIS BULLSHIT

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=70, val_shift_limit=40, p=0.9),

    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=(114,114,114)),

    A.ToTensorV2()
    ],bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_area=4
        ,min_visibility=0.1,
        check_each_transform=True
        )
    )

mosaic_transform = A.Compose([
    A.Mosaic(grid_yx=(2,2), target_size=[IMG_SIZE, IMG_SIZE]),
    ],bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_area=4
        ,min_visibility=0.1,
        check_each_transform=True
    )
)

letterbox = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114)),
], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_area=4
        ,min_visibility=0.1,
        check_each_transform=True
    ))

MOSAIC_PROB = 0.3
MIXUP = 0.15