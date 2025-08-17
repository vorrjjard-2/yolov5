import torch
from dataset import YOLODataset
import config

YoloV5Dataset = YOLODataset(
        split='train'
        ,transform=config.train_transform
    ) 