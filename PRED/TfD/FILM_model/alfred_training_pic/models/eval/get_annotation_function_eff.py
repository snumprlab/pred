#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 01:02:02 2021

@author: soyeonmin
"""

import pycocotools.mask as mask_util
import numpy as np
import os, sys
import cv2

import matplotlib
if sys.platform == "darwin":
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from PIL import Image

def get_annotation_instances(
    masks,
    step_num,
    label_ids,
):

    annos = []
    masks = [mask != 0.0 for mask in masks]
    masks = [mask.astype(np.uint8) for mask in masks]
    
    num_objs = len(masks)
    boxes = []
    objects_added = 0
    anno_id_counter = step_num * 1000  # force uniqueness
    img_id_counter = step_num
    for j in range(num_objs):
        mask = masks[j]
        #poly = polys[j]
        is_crowd = 0
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        bbox = mask_util.toBbox(rle)
        mask_area = int(mask_util.area(rle))
        if mask_area > 36:
            boxes.append(bbox)
            anno_id_counter += 1
            annotation = {
                "category_id": label_ids[j]+1,
                "segmentation": rle,
                "area": mask_area,
                "bbox": list(bbox),
                "iscrowd": is_crowd,
                #"mask": mask.tolist(),
                #"mask": [poly],
            }
            annotation['segmentation']['counts'] = annotation['segmentation']['counts'].decode('utf-8') 
            annos.append(annotation)
            objects_added += 1

    return annos