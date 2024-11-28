import os, sys
import matplotlib

os.environ["FILM_model_dir"]='FILM_model'
os.environ["TEACH_DIR"]=''
import sys
sys.path.insert(0,os.path.join(os.environ["TEACH_DIR"], os.environ["FILM_model_dir"]))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["DEPTH"] = "False"
#os.environ["DEPTH"] = "True"


if sys.platform == 'darwin':
	matplotlib.use("tkagg")

import torch
torch.multiprocessing.set_start_method('spawn',  force=True)# good solution !!!!
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
import time

import cv2
from torchvision import transforms
from PIL import Image
import skimage.morphology

from importlib import import_module
from collections import defaultdict
import json, pickle
from datetime import datetime

from arguments import get_args
#from envs import make_vec_envs
import envs.utils.pose as pu

from models.sem_mapping import Semantic_Mapping
from models.instructions_processed_LP.ALFRED_task_helper import determine_consecutive_interx
import alfred_utils.gen.constants as constants
#from models.semantic_policy.sem_map_model import UNetMulti

from main_class import ClassMain
from typing import List

from controller_without_sim import SemExp_ControllerWithoutSim
import copy
import skimage.morphology
import skimage.measure

from models.instructions_processed_LP.InitPickupMask import InitPickupMask


#class MainModel(TeachModel):
class FILMModel:
	def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
		#parser = argparse.ArgumentParser()
		#parser.add_argument("--seed", type=int, default=1, help="Random seed")
		#main_args = parser.parse_args(model_args)

		#Location of models

		#
		global args
		global envs
		args = get_args(model_args)
		args.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		args.skip_indices = {}
		self.args = args
		
		self.sem_exp = SemExp_ControllerWithoutSim(args, 0)
		self.main = ClassMain(args, self.sem_exp)
		self.episode_number = -1
		#Do until Init map and pose

		#Initalize map
		self.last_plan_act_and_preprocess = True
		self.plan_act_and_preprocess_accrued_actions = []

		#DELETE LATER!!
		#self.env = env
		#self.sem_exp.env = env
		self.main.sem_exp = self.sem_exp
		self.args.pickup_mask_enable = True
		self.just_stop=False
		self.need_to_be_initalized = True
		self.interactions = set(["ToggleObjectOn", "PourObject", "ToggleObjectOff", "OpenObject","CloseObject", 'SliceObject', "PickupObject", "PutObject"]) 

	def get_instance_metrics(self, instance_metrics):
		self.main.instance_metrics = instance_metrics

	#def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
	def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
		#start_pose_image = np.array(edh_history_images[-1])
		#start_pose_image = cv2.resize(start_pose_image, (300,300))
		start_pose_image = np.zeros((300,300)).astype('uint8')
		self.edh_history_images =  [cv2.resize(np.array(im), (300,300)) for im in edh_history_images]
		#Temporarily
		#self.sem_exp.depth =cv2.resize(depth, (300,300)) 
		self.episode_number +=1
		print("Episode number is ", self.episode_number)
		print("Just stop is ", self.just_stop)
		if self.episode_number ==0 or self.just_stop or not hasattr(self.main,'next_step_dict_s'):#If ended with exception in the first episode
			#_, self.main.infos, self.main.actions_dicts, self.main.return_actions = self.sem_exp.load_initial_scene(edh_instance, start_pose_image)
			
			if self.episode_number!= 0:
				self.main.write_log()
			
			_, infos, actions_dicts, return_actions = self.sem_exp.load_initial_scene(edh_instance, start_pose_image)
			#self.sem_exp.load_initial_scene()
			#pass #Already done in self.main
			#self.main.load_initial_scene_edh(edh_instance, start_pose_image)
		elif self.episode_number >0:
			#If this was called but not self.main.task_finish[0], then do this
			#if not(self.main.task_finish[0]):
				#self.main.start_again_if_finished1()
			#if not(self.main.task_finish[e]):
			#print("SEM EXP Log Before writing is  ", self.sem_exp.logs)
			self.main.write_log()
			self.main.task_finish[0] = True
			self.PPP_actions = []
			self.prev_action = "Stop"; self.sem_exp.prev_action = self.prev_action; self.last_success_special_treatment = None
			self.main.start_again_if_finished1()
			self.need_to_be_initalized = False
			

			_, infos, actions_dicts, return_actions = self.sem_exp.load_next_scene(edh_instance, start_pose_image)
			#print("SEM EXP Log is ", self.sem_exp.logs)
			#self.episode_number +=1
			#Do we need to do after_step 1 and 2 here?
			#Main.after_step1()
			#self.main.start_again_if_finished(edh_instance, start_pose_image)
			#Main.after_step3()
		if (infos is None) and (actions_dicts is None) and (return_actions is None):
			print("Fell into JUST STOP TRUE")
			self.PPP_actions = [('Meaningless', None, None), ('Stop_EmptyList', None, None)]
			if self.episode_number ==0:
				self.just_stop = True
			return True
		self.main.infos, self.main.actions_dicts, self.main.return_actions = infos, actions_dicts, return_actions

		print("Ran till here2")
		#self.PPP_actions = [('LookDown_15', None), ('LookDown_15', None), ('LookDown_15', None)]
		if self.sem_exp.camera_horizon == 0:
			self.PPP_actions = [('Meaningless', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None)] #BECAUSE OF self.PPP_actions = self.PPP_actions[1:], need to put one meaningless action
		elif self.sem_exp.camera_horizon == 30:
			self.PPP_actions = [('Meaningless', None, None), ('LookDown_30', None, None)] #BECAUSE OF self.PPP_actions = self.PPP_actions[1:], need to put one meaningless action
			self.looked_down_three_finished = False
		elif self.sem_exp.camera_horizon == 60:
			self.PPP_actions = [('Meaningless', None, None), ('RotateLeft_90', None, None), ('RotateRight_90', None, None)]
		elif self.sem_exp.camera_horizon == 90:
			self.PPP_actions = [('Meaningless', None, None), ('LookUp_30', None, None)]
		elif self.sem_exp.camera_horizon == -30:
			self.PPP_actions = [('Meaningless', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None)]
		elif self.sem_exp.camera_horizon == -60:
			self.PPP_actions = [('Meaningless', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None), ('LookDown_30', None, None)]
		#self.looked_down_three_finished = False
		self.sem_exp.print_log("Starting self.camera_horizon is ", self.sem_exp.camera_horizon)
		print("Ran till here3")
		#if simulator_horizon != self.sem_exp.camera_horizon:

		self.resetted_goal_with_look = False 
		self.prev_action =  self.sem_exp.prev_action = 'init'
		self.last_success_special_treatment = None
		self.FILM_main_step = 0
		self.sem_exp.prev_rgb = copy.deepcopy(start_pose_image)
		self.notPickupmask_yet = True
		self.just_stop = False
		self.looked_down_three_finished = False
		return True

	def convert2_teach_actions(self, action):
		teach_action_list = ['Stop', 'Move to', 'Forward', 'Backward', 'Turn Left', 'Turn Right', 'Look Up', 'Look Down', 'Pan Left', 'Pan Right', 'Move Up', 'Move Down', 'Double Forward', 'Double Backward', 'Navigation', 'Pickup', 'Place', 'Open', 'Close', 'ToggleOn', 'ToggleOff', 'Slice', 'Dirty', 'Clean', 'Fill', 'Empty', 'Pour', 'Break', 'BehindAboveOn', 'BehindAboveOff', 'OpenProgressCheck', 'SelectOid', 'SearchObject', 'Text', 'Speech', 'Beep']
		teach_dict = {'Stop': 'Stop',
						'MoveAhead_25': 'Forward',
						"RotateLeft_90":'Turn Left',
						"RotateRight_90":'Turn Right',
						"LookDown_30":'Look Down',
						"LookUp_0":'Pass',
						"LookDown_0":'Pass',
						"LookUp_30":'Look Up',
						"PickupObject":'Pickup',
						"PutObject": "Place",
						"OpenObject": 'Open',
						"CloseObject": 'Close',
						'SliceObject': 'Slice',
						"ToggleObjectOn": 'ToggleOn',
						"ToggleObjectOff": 'ToggleOff',
						"PourObject": "Pour"
						}
		return teach_dict[action]

	def get_coord_from_interaction_mask(self, outer_action, interaction_mask, size=300):
		if not(outer_action in self.interactions) and interaction_mask is None:
			return None
		elif outer_action in self.interactions and interaction_mask is None:
			return (0.5, 0.5)
		else:
			#if self.sem_exp.goal_name in ["CounterTop", "Desk", "DiningTable", 'BathtubBasin', 'Fridge', 'Dresser', 'CoffeeTable', 'Shelf', 'TVStand', 'Bed']:
			if 1==1:
				#Erode and choose a random point
				selem = skimage.morphology.disk(3)
				new_mask = skimage.morphology.binary_dilation(interaction_mask, selem)
				wheres = np.where(new_mask == True)
				#Select a random point
				np.random.seed(self.sem_exp.steps_taken)
				chosen = np.random.choice(len(wheres[0]))
				chosen_x = wheres[0][chosen]; chosen_y = wheres[1][chosen]
				return (float(chosen_x)/300, float(chosen_y)/300)


			#return list(np.array(interaction_mask.nonzero()).mean(axis=1) / float(size))
			connected_regions = skimage.morphology.label(interaction_mask, connectivity=2)
			connected_regions[np.where(interaction_mask == 0)] = -1
			unique_labels = sorted(list(set(connected_regions.flatten().tolist())))
			unique_labels = [u for u in unique_labels if u>-1]
			#Only get the ones that have value 1
			return_mask = np.zeros((300,300))
			lab_area = {lab:0 for lab in unique_labels}
			max_ar = 0
			largest_lab = None
			for lab in unique_labels:
				wheres = np.where(connected_regions == lab)
				lab_area[lab] = np.sum(len(wheres[0]))
				if lab_area[lab] > 100:
					max_ar = max(lab_area[lab], max_ar)
					if max_ar == lab_area[lab]:
						largest_lab = lab

			if not(largest_lab) is None:
				return_mask[np.where(connected_regions == largest_lab)] = 1

			#Finally get centroid
			M = skimage.measure.moments(return_mask)
			centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
			#if not(return_mask[int(centroid[0]), int(centroid[1])] == 1):
			#	pickle.dump(centroid, open('centroid.p', 'wb'))
			#	pickle.dump(return_mask, open('return_mask.p', 'wb'))
			#assert return_mask[int(centroid[0]), int(centroid[1])] == 1
			return (float(int(centroid[0]))/300, float(int(centroid[1]))/300)


	def get_pickup_mask(self, edh_instance, img):
		if self.args.pickup_mask_enable and self.notPickupmask_yet:
			print("Going through producing masks!")
			#self.sem_exp.print_log("EDH instance ID is ", edh_instance['instance_id'])
			#Set pickup mask
			PickUpClass = InitPickupMask(self.args, self.args.eval_split, edh_instance, self.edh_history_images, img)
			picked_up_mask, picked_up = PickUpClass.get_pickup_mask(self.sem_exp.seg.sem_seg_model_alfw_large, self.sem_exp.seg.sem_seg_model_alfw_small)
			self.sem_exp.picked_up_mask =  picked_up_mask
			self.sem_exp.picked_up = picked_up
			#Save into pictures
			if PickUpClass.last_pickup:
				self.sem_exp.print_log("Last Pickup for INIT PICKUP MASK")
			elif PickUpClass.last_place:
				self.sem_exp.print_log("Last Place for INIT PICKUP MASK")
			if PickUpClass.none:
				self.sem_exp.print_log("None for pick up mask outputted")
			self.notPickupmask_yet = False
			#Visualize: Just do later
			# if not(os.path.exists("mask_see/" + str(self.episode_number) + "/pick_up_mask")):
			# 	os.makedirs("mask_see/" + str(self.episode_number)+ "/pick_up_mask")

			# cv2.imwrite("mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "init_img.png", PickUpClass.driver_image_history[-1])
			# print("Wrote mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "init_img.png")
			if picked_up:
				#Save pictures
				# cv2.imwrite("mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "picked_up_img.png", PickUpClass.rgb)
				if PickUpClass.none:
					self.sem_exp.print_log("None for INIT PICKUP MASK")
				else:
					p = np.zeros((300,300, 3)); p[:, :, 0] = picked_up_mask; p[:, :, 1] = picked_up_mask; p[:, :, 2] = picked_up_mask
					picked_up_mask = p.astype('uint8') * 255
					# cv2.imwrite("mask_see/" + str(self.episode_number) + "/pick_up_mask/"+ "picked_up_mask.png", picked_up_mask)



	def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
		print("episode number is ", self.episode_number)
		if self.just_stop:
			return "Stop", None
		if len(self.PPP_actions)>=2 and self.PPP_actions[1][0] == "Stop_EmptyList":
			return "Stop", None

		#Pop from PPP_actions
		img = np.array(img)
		img = cv2.resize(img, (300,300))
		self.sem_exp.print_log("self.fails_cur is ", self.sem_exp.fails_cur)
		#Don't do picked up mask for now
		#Previous pick up mask location
		# if self.args.pickup_mask_enable and self.notPickupmask_yet:
		# 	print("Going through producing masks!")
		# 	self.sem_exp.print_log("EDH instance ID is ", edh_instance['instance_id'])
		# 	#Set pickup mask
		# 	PickUpClass = InitPickupMask(self.args, self.args.eval_split, edh_instance, img)
		# 	picked_up_mask, picked_up = PickUpClass.get_pickup_mask(self.sem_exp.seg.sem_seg_model_alfw_large, self.sem_exp.seg.sem_seg_model_alfw_small)
		# 	self.sem_exp.picked_up_mask =  picked_up_mask
		# 	self.sem_exp.picked_up = picked_up
		# 	#Save into pictures
		# 	if PickUpClass.last_pickup:
		# 		self.sem_exp.print_log("Last Pickup for INIT PICKUP MASK")
		# 	elif PickUpClass.last_place:
		# 		self.sem_exp.print_log("Last Place for INIT PICKUP MASK")
		# 	if PickUpClass.none:
		# 		self.sem_exp.print_log("None for pick up mask outputted")
		# 	self.notPickupmask_yet = False
		# 	#Visualize: Just do later
		# 	if not(os.path.exists("mask_see/" + str(self.episode_number) + "/pick_up_mask")):
		# 		os.makedirs("mask_see/" + str(self.episode_number)+ "/pick_up_mask")

		# 	cv2.imwrite("mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "init_img.png", PickUpClass.driver_image_history[-1])
		# 	print("Wrote mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "init_img.png")
		# 	if picked_up:
		# 		#Save pictures
		# 		cv2.imwrite("mask_see/" + str(self.episode_number)+ "/pick_up_mask/"+ "picked_up_img.png", PickUpClass.rgb)
		# 		if PickUpClass.none:
		# 			self.sem_exp.print_log("None for INIT PICKUP MASK")
		# 		else:
		# 			p = np.zeros((300,300, 3)); p[:, :, 0] = picked_up_mask; p[:, :, 1] = picked_up_mask; p[:, :, 2] = picked_up_mask
		# 			picked_up_mask = p.astype('uint8') * 255
		# 			cv2.imwrite("mask_see/" + str(self.episode_number) + "/pick_up_mask/"+ "picked_up_mask.png", picked_up_mask)


		self.PPP_actions = self.PPP_actions[1:]
		self.sem_exp.img = img

		#Temporarily
		#self.sem_exp.depth =cv2.resize(depth, (300,300)) 
		if self.looked_down_three_finished:
			self.sem_exp.semexp_step_internal_right_after_sim_step(self.last_success_special_treatment)
			self.sem_exp.print_log("FILM MODEL prev action was ", prev_action)
			self.sem_exp.print_log("camera horizon is ", self.sem_exp.camera_horizon) #Why is this 0
			#self.sem_exp.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			#assert self.sem_exp.camera_horizon == 45

		if self.looked_down_three_finished == False and self.PPP_actions ==[]:
			#GET PICKUP MASK AFTER THE AGENT LOOKS DOWN TO 60 DEGREES
			self.get_pickup_mask(edh_instance, img)
			self.sem_exp.semexp_step_internal_right_after_sim_step(self.last_success_special_treatment)
			self.sem_exp.print_log("camera horizon is ", self.sem_exp.camera_horizon) #Why is this 0
			#self.sem_exp.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			print("sem exp camera horizon is ", self.sem_exp.camera_horizon)
			#assert self.sem_exp.camera_horizon == 60
			if self.sem_exp.camera_horizon != 60:
				self.sem_exp.print_log("SEM EXP HORIZON NOT STARTING AT 60! INSTEAD STARTING AT ", self.sem_exp.camera_horizon)

			if self.need_to_be_initalized:
				self.main.obs = self.sem_exp.setup_scene_step0()
				self.main.infos = self.sem_exp.info; self.main.actions_dicts=self.sem_exp.actions_dict
				#self.main.obs = torch.tensor(np.expand_dims(self.main.obs, 0)).float(); self.infos = (self.infos, )
				self.main.load_initial_scene_edh(edh_instance,img)
			else:
				self.main.obs = self.sem_exp.setup_scene_step0()
				self.main.infos = self.sem_exp.info; self.main.actions_dicts=self.sem_exp.actions_dict
				#self.main.obs = torch.tensor(np.expand_dims(self.main.obs, 0)).float(); self.infos = (self.infos, )
				self.main.start_again_if_finished2() ####HERE NEEDS TO BE FIXED!
				self.main.after_step3()
			self.looked_down_three_finished = True
			print("camera horizon is ", self.sem_exp.camera_horizon) #Why is this 0
			#print("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			#assert self.sem_exp.camera_horizon == 45


		elif self.resetted_goal_with_look:
			if self.PPP_actions == []:
				self.main.after_step2()
				self.main.after_step3()
			self.resetted_goal_with_look = False

		elif self.PPP_actions ==[]:#This is when we communicate with main.py
			#Excute  PPP
			obs, info, goal_success, next_step_dict =  self.sem_exp.semexp_plan_act_and_preprocess_after_step(self.main.planner_inputs[0], self.main.goal_spotted_s[0])
			#Main.semexp_plan_act_and_preprocess_after_step = semexp_plan_act_and_preprocess_after_step #Do I need this? #Quickly check with step or somethin
			#Communicate with main
			if (self.need_to_be_initalized) and self.FILM_main_step == 0:
				self.main.after_step_taken_initial(obs, info, goal_success, next_step_dict) #HHHHHHERE! 
				self.FILM_main_step +=1
			#elif self.episode_number >0 and self.FILM_main_step == 0:
			#	#NEED TO PUT IN CODE HERE!
			#	self.FILM_main_step +=1
			else:
				self.main.after_step1(obs, info, goal_success, next_step_dict)
				self.PPP_actions = self.main.after_step1_reset() #THIS RETURNS actions
				# print("Reset ppp actions are ",self.PPP_actions )
				self.sem_exp.print_log("Reset ppp actions are ",self.PPP_actions )
				if self.PPP_actions !=[]:
					self.resetted_goal_with_look = True
					print("PPP actions is ", self.PPP_actions)
					self.prev_action = self.PPP_actions[0][0]; self.sem_exp.prev_action = self.prev_action; self.last_success_special_treatment = self.PPP_actions[0][2]
					print("Mask came from here ",)
					return self.convert2_teach_actions(self.PPP_actions[0][0]), None
				self.main.after_step2()

				if (hasattr(self.args, 'ignore_episodes_below') and self.episode_number < self.args.ignore_episodes_below)  or self.main.task_finish[0] or (self.sem_exp.task_type == "Breakfast" and self.sem_exp.steps_taken > 7):
					#self.main.start_again_if_finished1()
					self.PPP_actions = []
					self.prev_action = "Stop"; self.sem_exp.prev_action = self.prev_action; self.last_success_special_treatment = None
					return "Stop", None
				else:
					self.main.after_step3()
				self.FILM_main_step +=1

		#For the first three steps, look down 15 degrees		
		if self.sem_exp.steps_taken == 0:
			pass

		#if self.sem_exp.steps_taken ==3:
		#	self.sem_exp.camera_horizon = 45 #Do we need this? #Maybe useful in teach too?

		#if self.sem_exp.steps_taken <= 2:
		#	outer_action = "LookDown_15"; mask = None
			#Make sure self.sem_exp returns pose difference of 45 degrees instead of 0

		if 1 != 1:
			pass
		else:
			#Do everything done in main from line 697 to 695
			if self.PPP_actions == []:
				#Do some things to take care
				self.sem_exp.consecutive_steps = False
				# if self.sem_exp.consecutive_interaction_executed:
				# 	self.sem_exp.right_after_consecutive_interaction(self.prev_action)
				# 	pickle.dump(self.prev_action, open('temp_pickles/self_prev_action.p', 'wb'))
				# 	self.sem_exp.consecutive_interaction_executed = False

				#Do the real thing
				self.PPP_actions = self.sem_exp.semexp_plan_act_and_preprocess_before_step(self.main.planner_inputs[0], self.main.goal_spotted_s[0])
				self.sem_exp.print_log("PPP actions is ",self.PPP_actions )
				outer_action = self.PPP_actions[0][0]; mask = self.PPP_actions[0][1]; self.last_success_special_treatment = self.PPP_actions[0][2]
				if len(self.PPP_actions) > 1:
					self.sem_exp.consecutive_steps = True
					self.sem_exp.final_sidestep = False
			else:
				#Do the actions in PPP_actions
				outer_action = self.PPP_actions[0][0]; mask = self.PPP_actions[0][1]; self.last_success_special_treatment = self.PPP_actions[0][2]
				if len(self.PPP_actions) == 1:
					self.sem_exp.final_sidestep = True

		self.sem_exp.semexp_step_internal_right_before_sim_step()
		#Finally do the main mapping module before step
		self.prev_action = outer_action; self.sem_exp.prev_action = self.prev_action
		#print("outer action is ", outer_action)
		#print("PPP actions is ", self.PPP_actions)
		print("Mask came from here ", mask)
		#print("Action being outputted now: ", outer_action)
		return self.convert2_teach_actions(outer_action), self.get_coord_from_interaction_mask(outer_action, mask)

	def into_grid(self, ori_grid, grid_size):
		one_cell_size = math.ceil(240/grid_size)
		return_grid = torch.zeros(grid_size,grid_size)
		for i in range(grid_size):
			for j in range(grid_size):
				if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float())>0:
					return_grid[i,j] = 1
		return return_grid

	#def main_after_before_plan_act_and_preprocess():



