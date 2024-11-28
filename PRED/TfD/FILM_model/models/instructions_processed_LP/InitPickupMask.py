import os
#os.chdir('/Users/soyeonmin/Documents/Alexa/teach/')
#teach_dir = '/Users/soyeonmin/Documents/Alexa/teach/'
#DATA_FOLDER='/Volumes/untitled2/teach_download/'
#DATA_FOLDER = os.environ['DATA_DIR']

import sys
sys.path.append(os.path.join(os.environ["FILM_model_dir"]))
from init_pickup_mask_helper import init_pickup_mask_helper
import cv2
import pickle
#Repeat for all self.edh_instances and save pickedup next scene mask, image mask, and last image
#import ipdb
#import ipdb

from glob import glob
import json
import numpy as np


class InitPickupMask:
    def __init__(self, args, eval_split, edh_instance, edh_history_images, init_img):
        self.args = args
        self.edh_instance = edh_instance
        self.init_img = init_img
        #self.image_dir = DATA_FOLDER + "/images/" + eval_split + "/" + self.edh_instance['instance_id'].split('.')[0] 
        #pickle.dump(self.image_dir, open('im_dir.p', 'wb'))
        #ipdb.set_trace()
        #pickle.dump(self.paths, open('paths.p', 'wb'))
        self.driver_image_history = edh_history_images + [self.init_img]
        #ipdb.set_trace()
        self.none = False
        self.last_pickup = False
        self.last_place = False

    def open_teach_img(self, image_folder_dir, img_file_dir):
        return cv2.resize(cv2.imread(os.path.join(self.image_dir, img_file_dir)), (300,300))

    def get_pickup_mask(self, sem_seg_model_alfw_large, sem_seg_model_alfw_small):
        if len(self.edh_instance['driver_action_history'])>0:
            if self.edh_instance['driver_action_history'][-1]['action_name'] == "Pickup":
                self.last_pickup = True

            if self.edh_instance['driver_action_history'][-1]['action_name'] == "Place":
                self.last_pickup = True

            #if not(self.edh_instance['driver_action_history'][-1]['action_name'] in ['Pickup', 'Place']):

            init_helper = init_pickup_mask_helper(self.args)
            #image_dir = DATA_FOLDER + "images/valid_seen/" + self.edh_instance['instance_id'].split('.')[0] 
            #IMPORTANT: each image history is before taking that option
            #ASSERT THAT the last action is not pickup

            #Get the last picked up action instance that succeded
            last_picked_i = -1
            for i, action_dict in enumerate(self.edh_instance['driver_action_history']):
                if action_dict['action_name'] == "Pickup":
                    prev_rgb = self.driver_image_history[i]
                    img =  self.driver_image_history[i+1]
                    assert not(img is  None) and not(prev_rgb is None)
                    if init_helper._get_approximate_success(prev_rgb, img, "PickupObject"):
                        last_picked_i = i


            last_put = -1
            for i, action_dict in enumerate(self.edh_instance['driver_action_history']):
                if action_dict['action_name'] == "Place":
                    prev_rgb = self.driver_image_history[i]
                    img =  self.driver_image_history[i+1]
                    assert not(img is  None) and not(prev_rgb is None)
                    if init_helper._get_approximate_success(prev_rgb, img, "PutObject"):
                        last_put = i

            picked_up_mask = None
            picked_up = False
            case = None
            if last_picked_i ==-1:
                case = False
            elif last_put > last_picked_i :
                case=False
            else:
                case = True
                picked_up = True
                #get the largest one
                goal_name = self.edh_instance['driver_action_history'][last_picked_i]['oid'].split('|')[0]
                rgb = self.driver_image_history[-1]
                self.rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                assert not(self.rgb is None)
                init_helper.get_sem_pred(self.rgb, goal_name, sem_seg_model_alfw_large, sem_seg_model_alfw_small)
                picked_up_mask = init_helper.sem_seg_get_instance_mask_from_obj_type_largest_only(goal_name)
                #if picked_up_mask is None:
                #    ipdb.set_trace()
                if picked_up_mask is None:
                    self.none = True
        
            return picked_up_mask, picked_up
        else:
            return None, False
