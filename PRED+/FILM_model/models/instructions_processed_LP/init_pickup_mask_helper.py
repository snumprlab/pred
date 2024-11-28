#Copied from utils/control_helper
import cv2
import torch
import torchvision.transforms as T
import copy
import numpy as np
import argparse


import skimage.morphology
import sys
import os
teach_dir = os.environ['TEACH_DIR']

sys.path.append(os.path.join(teach_dir, os.environ['FILM_model_dir'], 'models/segmentation'))
from alfworld_mrcnn import load_pretrained_model 
import alfworld_constants



class init_pickup_mask_helper:
    def __init__(self, args):
        self.transform = T.Compose([T.ToTensor()])
        self.sem_seg_gpu =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #parser = argparse.ArgumentParser()
        #self.args = parser.parse_args()
        self.cuda = torch.cuda.is_available()
        self.with_mask_above_05 = True
        self.sem_seg_threshold_small =  0.5 #Just use smaller for picked up mask
        self.sem_seg_threshold_large =  0.8

        #Later replace these with the model defined in controller
        # self.sem_seg_gpu =  torch.device("cuda:" + str(args.sem_seg_gpu) if args.cuda else "cpu")
        # #LARGE
        # sem_seg_model_alfw_small = load_pretrained_model(teach_dir +'src/teach/inference/FILM_refactor_april1/models/segmentation/maskrcnn_alfworld/receps_lr5e-3_003.pth', torch.device("cuda:0" if args.cuda else "cpu"), 'recep')
        # sem_seg_model_alfw_small.eval()
        # sem_seg_model_alfw_small.to(torch.device("cuda:0" if args.cuda else "cpu"))                  

        # #SMALL            
        # sem_seg_model_alfw_small = load_pretrained_model(teach_dir +'src/teach/inference/FILM_refactor_april1/models/segmentation/maskrcnn_alfworld/objects_lr5e-3_005.pth', self.sem_seg_gpu, 'obj')
        # sem_seg_model_alfw_small.eval()
        # sem_seg_model_alfw_small.to(self.sem_seg_gpu)
        #sem_seg_model_alfw_small = sem_seg_model_alfw_small
        #sem_seg_model_alfw_small = sem_seg_model_alfw_large

        self.large = alfworld_constants.STATIC_RECEPTACLES
        self.large_objects2idx = {k:i for i, k in enumerate(self.large)}
        self.large_idx2large_object = {v:k for k,v in self.large_objects2idx.items()}

        self.small = alfworld_constants.OBJECTS_DETECTOR
        self.small_objects2idx = {k:i for i, k in enumerate(self.small)}
        self.small_idx2small_object = {v:k for k,v in self.small_objects2idx.items()}

        #self.cat_equate_dict = {'ButterKnife': 'Knife'}
        self.cat_equate_dict = {}
        


    def _get_approximate_success(self, prev_rgb, frame, action):
        wheres = np.where(prev_rgb != frame)
        wheres_ar = np.zeros(prev_rgb.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)
        if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
            success = True
        elif max_area > 100:
            success = True
        else:
            success = False
        return success

    def get_sem_pred(self, rgb, object_type, sem_seg_model_alfw_large, sem_seg_model_alfw_small):
        #1. First get dict
        self.get_instance_mask_seg_alfworld_both(rgb, object_type, sem_seg_model_alfw_large, sem_seg_model_alfw_small)

        #2. Get segmentation for map making
        #semantic_pred = self.segmentation_for_map()

        #3. visualize (get sem_vis)
        #if self.visualize or self.save_pictures:
        #    sem_vis = self.visualize_sem(rgb)


    def get_instance_mask_seg_alfworld_both(self, rgb, object_type, sem_seg_model_alfw_large, sem_seg_model_alfw_small):
        #rgb = copy.deepcopy(self.agent.event.frame)
        self.total_cat2idx = {object_type: 0}

        ims = [rgb]
        im_tensors = [self.transform(i).to(device=self.sem_seg_gpu) if self.cuda else self.transform(i) for i in ims]
        results_small = sem_seg_model_alfw_small(im_tensors)[0]        

        im_tensors = [self.transform(i).to(device=torch.device("cuda" if self.cuda else "cpu")) if self.cuda else self.transform(i) for i in ims] 
        results_large = sem_seg_model_alfw_small(im_tensors)[0]

        desired_classes_small = []
        desired_classes_large = []


        desired_goal_small = []
        desired_goal_large = []
        for cat_name in self.total_cat2idx:
            if not(cat_name in ["None", "fake"]):
                if cat_name in self.large:
                    large_class = self.large_objects2idx[cat_name]
                    desired_classes_large.append(large_class)
                    if cat_name == object_type:
                        desired_goal_large.append(large_class)

                elif cat_name in self.small:
                    small_class = self.small_objects2idx[cat_name]
                    desired_classes_small.append(small_class)
                    if cat_name == object_type:
                        desired_goal_small.append(small_class)
                else:
                    pass

        desired_goal_small = list(set(desired_goal_small))
        desired_goal_large = list(set(desired_goal_large))

        #FROM here
        indices_small = []     
        indices_large = []    

        for k in range(len(results_small['labels'])):
            if (
                results_small['labels'][k].item() in desired_classes_small
                and results_small['scores'][k] > self.sem_seg_threshold_small
            ):
                indices_small.append(k)


        for k in range(len(results_large['labels'])):
            if (
                results_large['labels'][k].item() in desired_classes_large
                and results_large['scores'][k] > self.sem_seg_threshold_large
            ):
                indices_large.append(k)


        #Done until here
        pred_boxes_small=results_small['boxes'][indices_small].detach().cpu()
        pred_classes_small=results_small['labels'][indices_small].detach().cpu()
        pred_masks_small=results_small['masks'][indices_small].squeeze(1).detach().cpu().numpy() #pred_masks[i] has shape (300,300)
        if self.with_mask_above_05:
            pred_masks_small = (pred_masks_small>0.5).astype(float)
        pred_scores_small = results_small['scores'][indices_small].detach().cpu()

        for ci in range(len(pred_classes_small)):
            if self.small_idx2small_object[int(pred_classes_small[ci].item())] in self.cat_equate_dict:
                cat = self.small_idx2small_object[int(pred_classes_small[ci].item())]
                pred_classes_small[ci] = self.small_objects2idx[self.cat_equate_dict[cat]]

        pred_boxes_large=results_large['boxes'][indices_large].detach().cpu()
        pred_classes_large=results_large['labels'][indices_large].detach().cpu()
        pred_masks_large=results_large['masks'][indices_large].squeeze(1).detach().cpu().numpy() #pred_masks[i] has shape (300,300)
        if self.with_mask_above_05:
            pred_masks_large = (pred_masks_large>0.5).astype(float)
        pred_scores_large = results_large['scores'][indices_large].detach().cpu()


        #Make the above into a dictionary
        self.segmented_dict  = {'small': {'boxes': pred_boxes_small,
                                            'classes': pred_classes_small,
                                            'masks': pred_masks_small,
                                            'scores': pred_scores_small
                                },
                                'large':{'boxes': pred_boxes_large,
                                        'classes': pred_classes_large,
                                        'masks': pred_masks_large,
                                        'scores': pred_scores_large}}
    def sem_seg_get_instance_mask_from_obj_type_largest_only(self, object_type):
        mask = np.zeros((300, 300))
        small_len = len(self.segmented_dict['small']['scores'])
        large_len = len(self.segmented_dict['large']['scores'])
        max_area = -1

        if object_type in self.cat_equate_dict:
            object_type = self.cat_equate_dict[object_type]

        if object_type in self.large_objects2idx:
            #Get the highest score
            for i in range(large_len):
                category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
                if category == object_type:
                    v = self.segmented_dict['large']['masks'][i]
                    score = self.segmented_dict['large']['scores'][i]
                    area = np.sum(self.segmented_dict['large']['masks'][i])
                    max_area = max(area, max_area)
                    if max_area == area:    
                        mask =  v.astype('float')

        else:
            for i in range(small_len):
                category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
                if category == object_type:
                    v = self.segmented_dict['small']['masks'][i]
                    score = self.segmented_dict['small']['scores'][i]
                    area = np.sum(self.segmented_dict['small']['masks'][i])
                    max_area = max(area, max_area)
                    if max_area == area:   
                        mask =  v.astype('float')

        if np.sum(mask) == 0:
            mask = None
        return mask
