import json
import config as config
import os

from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import cv2

def pre_index(annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)

    id_annotations = defaultdict(list)
    id_images = {}
    id_categories = {} 

    #ID TO ANNOTATIONS

    for item in annotations["annotations"]:
        img_id = item['image_id']
        id_annotations[img_id].append([item['category_id']] + item['bbox'])

    # ID TO IMAGE
    for item in annotations["images"]:
        id = item['id']
        id_images[id] = (item['file_name'], item['width'], item['height'])
    
    for item in annotations['categories']:
        category_id = item['id']
        id_categories[category_id] = item['name']
        
    return (id_annotations, id_images, id_categories)


def plot_image(raw_img, bboxes):
    raw_img = raw_img.permute(1, 2, 0)
    bboxes = denormalize_bboxes(bboxes, config.IMG_SIZE, config.IMG_SIZE)

    image = np.array(raw_img)
    fig, ax = plt.subplots(1)

    print(image.shape)

    ax.imshow(image)
    for bbox_coords in bboxes:
        rect = patches.Rectangle((bbox_coords[2], bbox_coords[3]),
                                bbox_coords[4], bbox_coords[5],
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    print(f'Displaying images with {len(bboxes)} boxes.' )

    plt.show()

### BOUNDING BOX HANDLING ###

def normalize_bboxes(raw_bboxes, W, H):
    norm_bboxes = []

    for box in raw_bboxes:
        c, x, y, w, h = box 
        x = (x + w/2) / W 
        y = (y + h/2) / H
        w = w / W
        h = h / H

        norm_bboxes.append([c,x,y,w,h])
    return norm_bboxes 

def denormalize_bboxes(raw_bboxes, W, H):
    denorm_boxes = []
    for box in raw_bboxes:
        i, c, x, y, w, h = box 
        x = (x - (w/2)) * W 
        y = (y - (h / 2)) * H 
        w = w * W 
        h = h * H 
        denorm_boxes.append([i, c, x, y, w, h])

    return denorm_boxes


