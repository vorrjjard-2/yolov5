import config
import numpy as np
import torch

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
        image = np.array(cv2.cvtColor(cv2.imread(os.path.join(config.DATA_ROOT, config.DATASET, self.split, raw_path)), cv2.COLOR_BGR2RGB))

        class_labels = [box[0] for box in bboxes]
        coords_only = [box[1:] for box in bboxes]

        if self.transform:
            augmented = self.transform(image=image, bboxes=coords_only, class_labels=class_labels)
            image = augmented["image"]
            coords_only = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            bboxes = [[cls] + list(coord) for cls, coord in zip(class_labels, coords_only)]

        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        print(bboxes)

        targets = torch.zeros((len(bboxes), 6))

        for i, bbox in enumerate(bboxes):
            targets[i, :] = torch.tensor(([index] + bbox)) 

        return (image, targets)
    
    def collate_fn(batch_input):

        collate_targets = torch.tensor([])

        images = [item[0] for item in batch_input]
        img_bboxes = [item[1][1:] for item in batch_input]

        for idx, setbboxes in enumerate(img_bboxes):
            if setbboxes:
                continue
            ELSE:




        collate_images = torch.stack(images, dim=0)

        return collate_images, collate_targets


if __name__ == '__main__':

    YoloV5Dataset = YOLODataset(
        split='train'
        ,transform=config.train_transform
    ) 

    test_img, test_bboxes, _ = YoloV5Dataset[4]
    print(test_bboxes)
    plot_image(test_img, test_bboxes)