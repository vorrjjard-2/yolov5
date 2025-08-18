import config
import numpy as np
import torch

import collections
from collections import defaultdict

from numpy import random 

from torch.utils.data import Dataset, DataLoader


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
                 ,mosaic=None
                 ,letterbox=None
                   ):
        super().__init__()
        self.split = split
        self.S = S
        self.transform = transform
        self.mosaic = mosaic
        self.letterbox = letterbox
        self.id_annotations, self.id_images, self.id_categories = pre_index(
            os.path.join('datasets', config.DATASET, self.split, '_annotations.coco.json')
        )
        self.epoch = 1

    def __len__(self):
        return len(self.id_annotations)
    
    def setepoch(self, E):
        self.epoch = E 

    def get_sample(self, index):
        raw_path, W, H = self.id_images[index]
        raw_bboxes = self.id_annotations[index]

        image = np.array(cv2.cvtColor(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, self.split, raw_path)), cv2.COLOR_BGR2RGB))
        bboxes = normalize_bboxes(raw_bboxes, W, H)

        class_labels = [box[0] for box in bboxes]
        bboxes = [box[1:] for box in bboxes]

        return image, bboxes, class_labels

    def metadata(self, index):
        meta_data = defaultdict(str)

        raw_path, W, H = self.id_images[index]
        raw_bboxes = self.id_annotations[index]

        meta_data['image'] = np.array(cv2.cvtColor(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, self.split, raw_path)), cv2.COLOR_BGR2RGB))
        bboxes = normalize_bboxes(raw_bboxes, W, H)

        meta_data['class_labels'] = [box[0] for box in bboxes]
        meta_data['bboxes'] = [box[1:] for box in bboxes]

        return meta_data

    def __getitem__(self, index):
        mixup = 0 
        raw_path, W, H = self.id_images[index]
        raw_bboxes = self.id_annotations[index]
       
        bboxes = normalize_bboxes(raw_bboxes, W, H)
        image = np.array(cv2.cvtColor(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, self.split, raw_path)), cv2.COLOR_BGR2RGB))

        class_labels = [box[0] for box in bboxes]
        coords_only = [box[1:] for box in bboxes]

        n = np.random.rand()
        if n <  config.MIXUP:

            img_b, bboxes_b, labels_b = self.get_sample(np.random.randint(len(self)))
            augmented_a = self.letterbox(image=image, bboxes=coords_only, class_labels=class_labels)
            augmented_b = self.letterbox(image=img_b, bboxes=bboxes_b, class_labels=labels_b)

            image, bboxes, class_labels = augmented_a["image"], augmented_a["bboxes"], augmented_a["class_labels"]
            img_b, bboxes_b, labels_b = augmented_b["image"], augmented_b["bboxes"], augmented_b["class_labels"]

            r = np.random.beta(32.0, 32.0)
            image = (image * r + img_b * (1 - r)).astype(np.uint8)
            coords_only = np.concat((bboxes, bboxes_b), 0)
            class_labels = np.concat((class_labels, labels_b), 0) 
            bboxes = [[cls] + list(coord) for cls, coord in zip(class_labels, coords_only)]

        elif mixup == 0 and n < config.MOSAIC_PROB:
            print('applying mosaic!')
            mosaic_metadata = [self.metadata(np.random.randint(len(self))) for _ in range(3)]
            augmented = self.mosaic(image=image, bboxes=coords_only, class_labels=class_labels, mosaic_metadata=mosaic_metadata)
            image = augmented["image"]
            coords_only = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            bboxes = [[cls] + list(coord) for cls, coord in zip(class_labels, coords_only)]

        if self.transform:
            augmented = self.transform(image=image, bboxes=coords_only, class_labels=class_labels)
            image = augmented["image"]
            coords_only = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            bboxes = [[cls] + list(coord) for cls, coord in zip(class_labels, coords_only)]

        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        targets = torch.zeros((len(bboxes), 6))

        for i, bbox in enumerate(bboxes):
            targets[i, :] = torch.tensor(([index] + bbox)) 

        return (image, targets)
    
    def collate_fn(batch_input):

        images = [item[0] for item in batch_input]
        collate_images = torch.stack(images, dim=0)

        img_bboxes = [item[1] for item in batch_input]
        collate_targets = []

        for idx, setbboxes in enumerate(img_bboxes):
            if setbboxes.numel() > 0: 
               
               setbboxes[:, 0] = idx
               collate_targets.append(setbboxes)
            else:
                continue

        if collate_targets:               # make sure it's not empty
            collate_targets = torch.cat(collate_targets, dim=0)
        else:
            collate_targets = torch.zeros((0,6))

        collate_images = torch.stack(images, dim=0)

        return collate_images, collate_targets
    
if __name__ == '__main__':

    YoloV5Dataset = YOLODataset(
        split='train'
        ,mosaic=config.mosaic_transform
        ,transform=config.train_transform
        ,letterbox=config.letterbox        
    ) 

    img, bboxes = YoloV5Dataset[6]

    plot_image(img, bboxes)

    train_loader = DataLoader(
        YoloV5Dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=YOLODataset.collate_fn
    )

    imgs, targets = next(iter(train_loader))

    assert imgs.size() == (config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert targets.size()[1] == 6 

    print('dataset - assertions passed.')
