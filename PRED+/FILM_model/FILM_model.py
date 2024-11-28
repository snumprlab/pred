import os, sys
import matplotlib

import sys
sys.path.insert(0,'.')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == 'darwin':
	matplotlib.use("tkagg")

import torch
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
from models.semantic_policy.sem_map_model import UNetMulti

from main_class import ClassMain
from typing import List

from controller_without_sim import SemExp_ControllerWithoutSim


#class MainModel(TeachModel):
class FILMModel:
	def __init__(self, process_index: int, num_processes: int, env, model_args: List[str]):
		#parser = argparse.ArgumentParser()
		#parser.add_argument("--seed", type=int, default=1, help="Random seed")
		#main_args = parser.parse_args(model_args)

		#Location of models

		#
		global args
		global envs
		args = get_args()
		args.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		args.device = torch.device("cuda:" + args.which_gpu if args.cuda else "cpu")
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
		self.env = env
		self.sem_exp.env = env
		self.main.sem_exp = self.sem_exp

	#def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
	def start_new_edh_instance(self, edh_instance, start_pose_image, edh_name=None):
		self.episode_number +=1
		if self.episode_number ==0:
			_, self.main.infos, self.main.actions_dicts, self.main.return_actions = self.sem_exp.load_initial_scene(edh_instance, start_pose_image)
			#self.sem_exp.load_initial_scene()
			#pass #Already done in self.main
			#self.main.load_initial_scene_edh(edh_instance, start_pose_image)
		elif self.episode_number >0:

			_, self.main.infos, self.main.actions_dicts, self.main.return_actions = self.sem_exp.load_next_scene(edh_instance, start_pose_image)
			#self.episode_number +=1
			#Do we need to do after_step 1 and 2 here?
			#Main.after_step1()
			#self.main.start_again_if_finished(edh_instance, start_pose_image)
			#Main.after_step3()

		#self.PPP_actions = [('LookDown_15', None), ('LookDown_15', None), ('LookDown_15', None)]
		self.PPP_actions = [('Meaningless', None, None), ('LookDown_15', None, None)] #BECAUSE OF self.PPP_actions = self.PPP_actions[1:], need to put one meaningless action
		self.looked_down_three_finished = False
		self.resetted_goal = False
		self.prev_action =  self.sem_exp.prev_action = 'init'
		self.last_success_special_treatment = None
		self.FILM_main_step = 0


	def get_instance_metrics(self, instance_metrics):
		self.main.instance_metrics = instance_metrics


	def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
		#Pop from PPP_actions
		self.sem_exp.print_log("self.fails_cur is ", self.sem_exp.fails_cur)
		self.PPP_actions = self.PPP_actions[1:]
		self.sem_exp.img = img

		if self.looked_down_three_finished:
			self.sem_exp.semexp_step_internal_right_after_sim_step(self.last_success_special_treatment)
			self.sem_exp.print_log("FILM MODEL prev action was ", prev_action)
			self.sem_exp.print_log("camera horizon is ", self.sem_exp.camera_horizon) #Why is this 0
			self.sem_exp.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			#assert self.sem_exp.camera_horizon == 45

		if self.looked_down_three_finished == False and self.PPP_actions ==[]:
			self.sem_exp.semexp_step_internal_right_after_sim_step(self.last_success_special_treatment)
			self.sem_exp.print_log("camera horizon is ", self.sem_exp.camera_horizon) #Why is this 0
			self.sem_exp.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			assert self.sem_exp.camera_horizon == 45

			if self.episode_number ==0:
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
			print("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon']) #WHy is this 60
			#assert self.sem_exp.camera_horizon == 45


		elif self.resetted_goal:
			if self.PPP_actions == []:
				self.main.after_step2()
				self.main.after_step3()

		elif self.PPP_actions ==[]:#This is when we communicate with main.py
			#Excute  PPP
			obs, info, goal_success, next_step_dict =  self.sem_exp.semexp_plan_act_and_preprocess_after_step(self.main.planner_inputs[0], self.main.goal_spotted_s[0])
			#Main.semexp_plan_act_and_preprocess_after_step = semexp_plan_act_and_preprocess_after_step #Do I need this? #Quickly check with step or somethin
			#Communicate with main
			if self.episode_number == 0 and self.FILM_main_step == 0:
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
					self.resetted_goal = True
					print("PPP actions is ", self.PPP_actions)
					self.prev_action = self.PPP_actions[0]; self.sem_exp.prev_action = self.prev_action; self.last_success_special_treatment = self.PPP_actions[0][2]
					print("Mask came from here ",)
					return self.PPP_actions[0], None
				self.main.after_step2()

				if self.main.task_finish[0]:
					self.main.start_again_if_finished1()
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
		return outer_action, mask

	def into_grid(self, ori_grid, grid_size):
		one_cell_size = math.ceil(240/grid_size)
		return_grid = torch.zeros(grid_size,grid_size)
		for i in range(grid_size):
			for j in range(grid_size):
				if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float())>0:
					return_grid[i,j] = 1
		return return_grid

	#def main_after_before_plan_act_and_preprocess():



