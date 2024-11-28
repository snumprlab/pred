import os, sys
import matplotlib

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
from models.instructions_processed_LP.object_state import determine_consecutive_interx
import alfred_utils.gen.constants as constants
from models.semantic_policy.sem_map_model import UNetMulti


def into_grid(ori_grid, grid_size):
    one_cell_size = math.ceil(240/grid_size)
    return_grid = torch.zeros(grid_size,grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float())>0:
                return_grid[i,j] = 1
    return return_grid



class ClassMain:
	#No real init
	#These are all init
	def __init__(self,args, envs):
	#self.args = get_args()
		self.args = args
		self.envs = envs
		#self.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		#self.args.dn = self.dn
		self.dn = self.args.dn
		if self.args.set_dn!="":
			self.args.dn = self.args.set_dn
			self.dn =  self.args.set_dn
		print("dn is ", self.dn)
		self.step = 0

		if not os.path.exists("results/logs"):
			os.makedirs("results/logs",exist_ok=True)
		if not os.path.exists("results/leaderboard"):
			os.makedirs("results/leaderboard",exist_ok=True)
		if not os.path.exists("results/successes"):
			os.makedirs("results/successes",exist_ok=True)
		if not os.path.exists("results/fails"):
			os.makedirs("results/fails",exist_ok=True)
		if not os.path.exists("results/analyze_recs"):
			os.makedirs("results/analyze_recs",exist_ok=True)        

		self.completed_episodes = []
		
		#Disable skip indices for now
		self.skip_indices = {}
		# self.skip_indices ={}
		# if self.args.exclude_list!="":
		# 	if self.args.exclude_list[-2:] == ".p": 
		# 		self.skip_indices = pickle.load(open(self.args.exclude_list, 'rb'))
		# 		self.skip_indices = {int(s):1 for s in self.skip_indices}
		# 	else:
		# 		self.skip_indices = [a for a in self.args.exclude_list.split(',')]
		# 		self.skip_indices = {int(s):1 for s in self.skip_indices}
		# self.args.skip_indices = self.skip_indices
		self.actseqs = []
		self.all_completed = [False] * self.args.num_processes
		self.successes = []; self.failures = []  
		self.analyze_recs = []
		self.traj_number =[0] *self.args.num_processes

		np.random.seed(self.args.seed)
		torch.manual_seed(self.args.seed)

		if self.args.cuda:
			torch.cuda.manual_seed(self.args.seed)

		self.large_objects2idx = {obj:i for i, obj in enumerate(constants.map_save_large_objects)}
		self.all_objects2idx = {o:i for i, o in enumerate(constants.map_all_objects)}
		self.softmax = nn.Softmax(dim=1)


		# Logging and loss variables
		self.num_scenes = self.args.num_processes
		self.num_episodes = [0] * self.args.num_processes
		for e in range(self.args.from_idx, self.args.to_idx):  
			remainder = e %self.args.num_processes
			self.num_episodes[remainder] +=1
		
		self.device = self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.args.use_sem_policy:
			self.Unet_model = UNetMulti((240,240), num_sem_categories=24).to(device=self.device)
			sd = torch.load('FILM_model/models/semantic_policy/best_model_multi.pt', map_location = self.device)
			self.Unet_model.load_state_dict(sd)
			del sd

		self.finished = np.zeros((self.args.num_processes))
		self.wait_env = np.zeros((self.args.num_processes))


		# Starting environments
		#torch.set_num_threads(1)
		#envs = make_vec_envs(args)
		self.fails = [0] * self.num_scenes
		self.prev_cns = [None] * self.num_scenes
		self.returned   = False ######################3
		
	def load_initial_scene_edh(self,edh_instance, image):
		#self.obs, self.infos, self.actions_dicts, self.return_actions = self.envs.load_initial_scene(edh_instance, image)
		#print("return actions are ", self.return_actions)
		#assert self.return_actions == []
		self.returned   = False  #####################
		self.return_actions == []
		self.obs = torch.tensor(np.expand_dims(self.obs, 0)).float().to(self.device); self.infos = (self.infos, ); self.actions_dicts= (self.actions_dicts, )
		#pickle.dump(self.obs, open('temp_pickles/load_initial_scene_obs.p', 'wb')); pickle.dump(self.infos, open('temp_pickles/load_initial_scene_infos.p', 'wb')); pickle.dump(self.actions_dicts, open('temp_pickles/load_initial_scene_actions_dicts.p', 'wb'))
		self.second_objects = []; self.list_of_actions_s = []; self.task_types = []; self.whether_sliced_s= [] 
		for e in range(self.args.num_processes):
			self.second_object = self.actions_dicts[e]['second_object']
			self.list_of_actions = self.actions_dicts[e]['list_of_actions']
			self.task_type = self.actions_dicts[e]['task_type']
			self.sliced = self.actions_dicts[e]['sliced']
			self.second_objects.append(self.second_object); self.list_of_actions_s.append(self.list_of_actions); self.task_types.append(self.task_type); self.whether_sliced_s.append(self.sliced)

		self.task_finish = [False] * self.args.num_processes
		self.first_steps = [True] * self.args.num_processes
		self.num_steps_so_far = [0] * self.args.num_processes
		self.load_goal_pointers = [0] * self.args.num_processes
		self.list_of_actions_pointer_s = [0] * self.args.num_processes
		self.goal_spotted_s = [False] * self.args.num_processes  
		self.list_of_actions_pointer_s = [0] * self.args.num_processes
		self.goal_logs = [[] for i in range(self.args.num_processes)]
		self.goal_cat_before_second_objects = [None] * self.args.num_processes
	  
		self.do_not_update_cat_s = [None] * self.args.num_processes
		self.wheres_delete_s = [np.zeros((240,240))] *self.args.num_processes
		
		#self.args.num_sem_categories = 1 + 1 + 1 + 5 * self.args.num_processes
		self.args.num_sem_categories = 12
		if self.args.use_sem_policy:
			self.args.num_sem_categories = self.args.num_sem_categories + 23
		self.obs = torch.tensor(self.obs).to(self.device)
		
		torch.set_grad_enabled(False)

		# Initialize map variables
		### Full map consists of multiple channels containing the following:
		### 1. Obstacle Map
		### 2. Exploread Area
		### 3. Current Agent Location
		### 4. Past Agent Locations
		### 5,6,7,.. : Semantic Categories
		self.nc = self.args.num_sem_categories + 4 # num channels

		# Calculating full and local map sizes
		self.map_size = self.args.map_size_cm // self.args.map_resolution
		self.full_w, self.full_h = self.map_size, self.map_size
		self.local_w, self.local_h = int(self.full_w / self.args.global_downscaling), \
						   int(self.full_h / self.args.global_downscaling)

		# Initializing full and local map
		self.full_map = torch.zeros(self.num_scenes, self.nc, self.full_w, self.full_h).float().to(self.device)
		self.local_map = torch.zeros(self.num_scenes, self.nc, self.local_w, self.local_h).float().to(self.device)

		# Initial full and local pose
		self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
		self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)

		# Origin of local map
		self.origins = np.zeros((self.num_scenes, 3))

		# Local Map Boundaries
		self.lmb = np.zeros((self.num_scenes, 4)).astype(int)

		### Planner pose inputs has 7 dimensions
		### 1-3 store continuous global agent location
		### 4-7 store local map boundaries
		self.planner_pose_inputs = np.zeros((self.num_scenes, 7))

		#BEFORE FUNCTIONS

		#AFTER FUNCTIONS

		self.init_map_and_pose()

		# slam
		self.sem_map_module = Semantic_Mapping(self.args).to(self.device)
		self.sem_map_module.eval()
		self.sem_map_module.set_view_angles([45] * self.args.num_processes)

		# Predict semantic map from frame 1
		self.poses = torch.from_numpy(np.asarray(
				[self.infos[env_idx]['sensor_pose'] for env_idx in range(self.num_scenes)])
				).float().to(self.device)
		# print("Poses is ", self.poses)
		
		_, self.local_map, _, self.local_pose, self.translated = \
			self.sem_map_module(self.obs, self.poses, self.local_map, self.local_pose)

		# Compute Global policy input
		locs = self.local_pose.cpu().numpy()

		for e in range(self.num_scenes):
			r, c = locs[e, 1], locs[e, 0]
			loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
							int(c * 100.0 / self.args.map_resolution)]

			self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.


		#For now
		self.global_goals = []
		for e in range(self.num_scenes):
			# np.random.seed(e); c1 = np.random.choice(self.local_w)
			# np.random.seed(e + 1000); c2 = np.random.choice(self.local_h)
			# self.global_goals.append((c1,c2))
			np.random.seed(e); c1 = 120
			np.random.seed(e + 1000); c2 = 120
			self.global_goals.append((c1,c2))			

		self.goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]

		for e in range(self.num_scenes):
			self.goal_maps[e][self.global_goals[e][0], self.global_goals[e][1]] = 1
			
		self.newly_goal_set = False
		self.planner_inputs = [{} for e in range(self.num_scenes)]
		for e, p_input in enumerate(self.planner_inputs):
			p_input['newly_goal_set'] =self.newly_goal_set
			p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
			p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
			p_input['pose_pred'] = self.planner_pose_inputs[e]
			p_input['goal'] = self.goal_maps[e] 
			p_input['new_goal'] = 1
			p_input['found_goal'] = 0
			p_input['wait'] = self.finished[e]
			p_input['list_of_actions'] = self.list_of_actions_s[e]
			p_input['list_of_actions_pointer'] = self.list_of_actions_pointer_s[e]
			p_input['consecutive_interaction'] = None
			p_input['consecutive_target'] = None
   
			p_input['class_map'] =self.local_map[e, 4:, :, :].cpu().numpy() ##DTA
			if self.args.visualize or self.args.print_images:
				self.local_map[e, -1, :, :] = 1e-5
				p_input['sem_map_pred'] = self.local_map[e, 4:, :,
					:].argmax(0).cpu().numpy()

		#Then first step is taken
		#return planner_inputs, goal_spotted_s

	def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
		loc_r, loc_c = agent_loc
		self.local_w, self.local_h = local_sizes
		self.full_w, self.full_h = full_sizes

		if self.args.global_downscaling > 1:
			gx1, gy1 = loc_r - self.local_w // 2, loc_c - self.local_h // 2
			gx2, gy2 = gx1 + self.local_w, gy1 + self.local_h
			if gx1 < 0:
				gx1, gx2 = 0, self.local_w
			if gx2 > self.full_w:
				gx1, gx2 = full_w - self.local_w, self.full_w

			if gy1 < 0:
				gy1, gy2 = 0, self.local_h
			if gy2 > self.full_h:
				gy1, gy2 = full_h - self.local_h, self.full_h
		else:
			gx1, gx2, gy1, gy2 = 0, self.full_w, 0, self.full_h

		return [gx1, gx2, gy1, gy2]

	def init_map_and_pose(self):
		self.full_map.fill_(0.)
		self.full_pose.fill_(0.)
		self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

		locs = self.full_pose.cpu().numpy()
		self.planner_pose_inputs[:, :3] = locs
		for e in range(self.num_scenes):
			r, c = locs[e, 1], locs[e, 0]
			loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
							int(c * 100.0 / self.args.map_resolution)]

			self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

			self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
											  (self.local_w, self.local_h),
											  (self.full_w, self.full_h))

			self.planner_pose_inputs[e, 3:] = self.lmb[e]
			self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
						  self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

		for e in range(self.num_scenes):
			self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
			self.local_pose[e] = self.full_pose[e] - \
							torch.from_numpy(self.origins[e]).to(self.device).float()

	def init_map_and_pose_for_env(self, e):
		self.full_map[e].fill_(0.)
		self.full_pose[e].fill_(0.)
		self.full_pose[e, :2] = self.args.map_size_cm / 100.0 / 2.0

		locs = self.full_pose[e].cpu().numpy()
		self.planner_pose_inputs[e, :3] = locs
		r, c = locs[1], locs[0]
		loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
						int(c * 100.0 / self.args.map_resolution)]

		self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

		self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
										  (self.local_w, self.local_h),
										  (self.full_w, self.full_h))

		self.planner_pose_inputs[e, 3:] = self.lmb[e]
		self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
					  self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

		self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
		self.local_pose[e] = self.full_pose[e] - \
						torch.from_numpy(self.origins[e]).to(self.device).float()

	#def after_step_taken_initial(self, obs, rew, done, infos, goal_success_s, next_step_dict_s):
	def after_step_taken_initial(self, obs, infos, goal_success_s, next_step_dict_s):
		#obs, rew, done, infos, goal_success_s, next_step_dict_s = self.envs.plan_act_and_preprocess(self.planner_inputs[0], self.goal_spotted_s[0])
		self.step +=1
		self.obs, self.infos, self.goal_success_s, self.next_step_dict_s = obs, infos, goal_success_s, next_step_dict_s
		self.obs = torch.tensor(np.expand_dims(self.obs, 0)).float().to(self.device); self.infos = (self.infos, ); self.goal_success_s = (self.goal_success_s, ); self.next_step_dict_s = (self.next_step_dict_s,)
		self.goal_success_s = list(self.goal_success_s)
		self.view_angles = []
		for e in range(self.num_scenes):
			self.next_step_dict = self.next_step_dict_s[e]
			self.view_angle = self.next_step_dict['view_angle']
			self.view_angles.append(self.view_angle)
			
			self.fails[e] += self.next_step_dict['fails_cur']
			
		self.sem_map_module.set_view_angles(self.view_angles)
		
		
		self.consecutive_interaction_s, self.target_instance_s = [None]*self.num_scenes, [None]*self.num_scenes
		for e in range(self.num_scenes):
			self.num_steps_so_far[e] = self.next_step_dict_s[e]['steps_taken']
			self.first_steps[e] = False
			# if self.goal_success_s[e]: 
			# 	if self.list_of_actions_pointer_s[e] == len(self.list_of_actions_s[e]) -1:
			# 		self.all_completed[e] = True
			# 	else:
			# 		self.list_of_actions_pointer_s[e] +=1
			# 		self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
					
			# 		self.reset_goal_true_false = [False]* self.num_scenes
			# 		self.reset_goal_true_false[e] = True
					
						
			# 		#If consecutive interactions, 
			# 		returned, self.target_instance_s[e] = determine_consecutive_interx(self.list_of_actions_s[e], self.list_of_actions_pointer_s[e]-1, self.whether_sliced_s[e])
			# 		if returned:
			# 			self.consecutive_interaction_s[e] = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][1]

			# 		self.infos, return_actions  = self.envs.reset_goal(self.reset_goal_true_false[e], self.goal_name, self.consecutive_interaction_s[e])
			# 		self.infos = (self.infos, )
		
		

		torch.set_grad_enabled(False)
		#spl_per_category = defaultdict(list)
		#success_per_category = defaultdict(list)

		#####################################
		#Beginning of for loop, until plan act and preprocess
		#########################################
		if sum(self.finished) == self.args.num_processes:
			print("all finished")
			if self.args.leaderboard and self.args.test:
				if self.args.test_seen:
					self.add_str = "seen"
				else:
					self.add_str = "unseen"
				pickle.dump(self.actseqs, open("results/leaderboard/actseqs_test_" + self.add_str + "_" + self.dn + ".p", "wb"))
			
			
			#raise Exception("Finished!")
			return True

		self.l_step = self.step % self.args.num_local_steps
		
		# ------------------------------------------------------------------
		# Semantic Mapping Module
		self.poses = torch.from_numpy(np.asarray(
					[self.infos[env_idx]['sensor_pose'] for env_idx
					 in range(self.num_scenes)])
				).float().to(self.device)
		# print("Poses is ", self.poses)

		_, self.local_map, _, self.local_pose, self.translated = self.sem_map_module(self.obs, self.poses, self.local_map, self.local_pose, build_maps = True, no_update = False)

		locs = self.local_pose.cpu().numpy()
		self.planner_pose_inputs[:, :3] = locs + self.origins
		self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
		for e in range(self.num_scenes):
			r, c = locs[e, 1], locs[e, 0]
			loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
							int(c * 100.0 / self.args.map_resolution)]
			self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

		for e in range(self.num_scenes):
			if not(self.do_not_update_cat_s[e] is None):
				cn = self.do_not_update_cat_s[e] + 4
				self.local_map[e, cn, :, :] = torch.zeros(self.local_map[0, 0, :, :].shape)

		for e in range(self.num_scenes):
            # remove the first object's pcd
			if len(self.second_objects[e]) > 0 and self.list_of_actions_pointer_s[e] < len(self.second_objects[e]):
				if self.second_objects[e][self.list_of_actions_pointer_s[e]]:
					cn = self.infos[e]['goal_cat_id'] + 4
					first_object_local_map = skimage.morphology.binary_dilation(
                        (self.translated.cpu().numpy()[e, cn, :, :] > 0).astype(int),
                        skimage.morphology.disk(4)
                    ).astype(int)
					self.wheres_delete_s[e][first_object_local_map > 0] = 1
					self.second_objects[e][self.list_of_actions_pointer_s[e]] = False
     
			if self.args.delete_from_map_after_move_until_visible and (self.next_step_dict_s[e]['move_until_visible_cycled'] or self.next_step_dict_s[e]['delete_lamp']):
				self.ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
				#Get the label that is closest to the current goal
				cn = self.infos[e]['goal_cat_id'] + 4

				start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.planner_pose_inputs[e]
				gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
				r, c = start_y, start_x
				start = [int(r * 100.0/self.args.map_resolution - gx1),
						 int(c * 100.0/self.args.map_resolution - gy1)]
				map_pred = np.rint(self.local_map[e, 0, :, :].cpu().numpy())
				assert self.local_map[e, 0, :, :].shape[0] == 240
				start = pu.threshold_poses(start, map_pred.shape)

				lm = self.local_map[e, cn, :, :].cpu().numpy()
				lm = (lm>0).astype(int)
				lm = skimage.morphology.binary_dilation(lm, skimage.morphology.disk(4))
				lm = lm.astype(int)
				connected_regions = skimage.morphology.label(lm, connectivity=2)
				unique_labels = [i for i in range(0, np.max(connected_regions)+1)]
				min_dist = 1000000000
				for lab in unique_labels:
					wheres = np.where(connected_regions == lab)
					center = (int(np.mean(wheres[0])), int(np.mean(wheres[1])))
					dist_pose = math.sqrt((start[0] -center[0])**2 + (start[1] -center[1])**2)
					min_dist = min(min_dist, dist_pose)
					if min_dist == dist_pose:
						min_lab = lab

				#Delete that label
				self.wheres_delete_s[e][np.where(connected_regions == min_lab)] = 1

				
		for e in range(self.num_scenes):
			cn = self.infos[e]['goal_cat_id'] + 4
			self.wheres = np.where(self.wheres_delete_s[e])
			self.local_map[e, cn, :, :][self.wheres] = 0.0


		# ------------------------------------------------------------------

		# ------------------------------------------------------------------
		# Semantic Policy
		self.newly_goal_set = False
		if self.l_step == self.args.num_local_steps - 1:
			self.newly_goal_set = True
			for e in range(self.num_scenes):
				#if self.wait_env[e] == 1: # New episode
				#	self.wait_env[e] = 0.

				self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
					self.local_map[e]
				self.full_pose[e] = self.local_pose[e] + \
							   torch.from_numpy(self.origins[e]).to(self.device).float()

				locs = self.full_pose[e].cpu().numpy()
				r, c = locs[1], locs[0]
				loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
								int(c * 100.0 / self.args.map_resolution)]

				self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
												  (self.local_w, self.local_h),
												  (self.full_w, self.full_h))

				self.planner_pose_inputs[e, 3:] = self.lmb[e]
				self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
							  self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

				self.local_map[e] = self.full_map[e, :,
							   self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
				self.local_pose[e] = self.full_pose[e] - \
								torch.from_numpy(self.origins[e]).to(self.device).float()


			locs = self.local_pose.cpu().numpy()
			
			for e in range(self.num_scenes):
				self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
				if  self.args.use_sem_policy:
					
					#Just reconst the common map save objects
					self.map_reconst = torch.zeros((4+len(self.large_objects2idx),240,240))
					self.map_reconst[:4] = self.local_map[e][:4]
					self.test_see = {}
					self.map_reconst[4+self.large_objects2idx['SinkBasin']] = self.local_map[e][4+1]
					self.test_see[1] = 'SinkBasin'

					start_idx = 2
					for cat, catid in self.large_objects2idx.items():
						if not (cat =='SinkBasin'):
							self.map_reconst[4+self.large_objects2idx[cat]] = self.local_map[e][4+start_idx]
							self.test_see[start_idx] = cat
							start_idx +=1

					if self.local_map[e][0][120,120] == 0:
						mask = np.zeros((240,240))
						connected_regions = skimage.morphology.label(1-self.local_map[e][0].cpu().numpy(), connectivity=2)
						connected_lab = connected_regions[120,120]
						mask[np.where(connected_regions==connected_lab)] = 1
						mask[np.where(skimage.morphology.binary_dilation(self.local_map[e][0].cpu().numpy(), skimage.morphology.square(4)))] = 1
					else:
						dilated = skimage.morphology.binary_dilation(self.local_map[e][0].cpu().numpy(), skimage.morphology.square(4))
						mask = skimage.morphology.convex_hull_image(dilated).astype(float)
					self.mask_grid = into_grid(torch.tensor(mask), 8).cpu()
					self.where_ones = len(torch.where(self.mask_grid)[0])
					self.mask_grid = self.mask_grid.repeat(73,1).view(73, -1).numpy()

					if  self.goal_name in self.all_objects2idx and self.next_step_dict_s[e]['steps_taken'] >= 30:

						pred_probs = self.Unet_model(self.map_reconst.unsqueeze(0).to(self.device))
						pred_probs = pred_probs.view(73, -1)
						pred_probs = self.softmax(pred_probs).cpu().numpy()

						pred_probs = (1-self.args.explore_prob) * pred_probs + self.args.explore_prob * self.mask_grid * 1/ float(self.where_ones)

						#Now sample from pred_probs according to goal idx
						self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
						if self.goal_name =='FloorLamp':
							pred_probs = pred_probs[self.all_objects2idx[self.goal_name]] + pred_probs[self.all_objects2idx['DeskLamp']]
							pred_probs = pred_probs/2.0
						else:
							pred_probs = pred_probs[self.all_objects2idx[self.goal_name]]


					else:
						pred_probs =  self.mask_grid[0] * 1/ float(self.where_ones)

					

					
					if self.args.explore_prob==1.0:
						self.mask_wheres = np.where(mask.astype(float))
						np.random.seed(self.next_step_dict_s[e]['steps_taken'])
						s_i= np.random.choice(len(self.mask_wheres[0]))
						x_240, y_240 = self.mask_wheres[0][s_i], self.mask_wheres[1][s_i]

					else:
						#Now sample one index
						np.random.seed(self.next_step_dict_s[e]['steps_taken'])
						pred_probs = pred_probs.astype('float64')
						pred_probs = pred_probs.reshape(64)
						pred_probs = pred_probs/ np.sum(pred_probs)

						chosen_cell = np.random.multinomial(1, pred_probs.tolist())
						chosen_cell = np.where(chosen_cell)[0][0]                        
						chosen_cell_x = int(chosen_cell/8)
						chosen_cell_y = chosen_cell %8
						
						
						#Sample among this mask
						self.mask_new = np.zeros((240,240))
						self.mask_new[chosen_cell_x*30:chosen_cell_x*30+30, chosen_cell_y*30: chosen_cell_y*30 + 30] = 1 
						self.mask_new = self.mask_new * mask
						if np.sum(self.mask_new) == 0:
							np.random.seed(self.next_step_dict_s[e]['steps_taken'])
							chosen_i = np.random.choice(len(np.where(mask)[0]))
							x_240 = np.where(mask)[0][chosen_i]
							y_240 = np.where(mask)[1][chosen_i]

						else:
							np.random.seed(self.next_step_dict_s[e]['steps_taken'])
							chosen_i = np.random.choice(len(np.where(self.mask_new)[0]))
							x_240 = np.where(self.mask_new)[0][chosen_i]
							y_240 = np.where(self.mask_new)[1][chosen_i]
						


					self.global_goals[e] = [x_240, y_240]
					self.test_goals= np.zeros((240,240))
					self.test_goals[x_240,y_240]=1

							

		# ------------------------------------------------------------------
	
	
		# ------------------------------------------------------------------
		# Take action and get next observation
		self.found_goal = [0 for _ in range(self.num_scenes)]
		self.goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]

		for e in range(self.num_scenes):
			self.goal_maps[e][self.global_goals[e][0], self.global_goals[e][1]] = 1


		for e in range(self.num_scenes):
			self.ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
			cn = self.infos[e]['goal_cat_id'] + 4
			self.prev_cns[e] = cn
			self.cur_goal_sliced = self.next_step_dict_s[e]['current_goal_sliced']

			if self.local_map[e, cn, :, :].sum() != 0.: 
				ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
				cat_semantic_map = self.local_map[e, cn, :, :].cpu().numpy()
				self.cat_semantic_scores = cat_semantic_map 

				self.cat_semantic_scores[self.cat_semantic_scores > 0] = 1.
				self.wheres = np.where(self.wheres_delete_s[e])
				self.cat_semantic_scores[self.wheres] = 0
				if np.sum(self.cat_semantic_scores) !=0:
					self.goal_maps[e] = self.cat_semantic_scores

				if np.sum(self.cat_semantic_scores) !=0:
					self.found_goal[e] = 1
					self.goal_spotted_s[e] = True
				else:
					if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
						self.found_goal[e] = 0
						self.goal_spotted_s[e] = False
			else:
				if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
					self.found_goal[e] = 0
					self.goal_spotted_s[e] = False



		self.planner_inputs = [{} for e in range(self.num_scenes)]
		for e, p_input in enumerate(self.planner_inputs):
			p_input['newly_goal_set'] =self.newly_goal_set
			p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
			p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
			p_input['pose_pred'] = self.planner_pose_inputs[e]
			p_input['goal'] = self.goal_maps[e] 
			p_input['new_goal'] = self.l_step == self.args.num_local_steps - 1
			p_input['found_goal'] = self.found_goal[e]
			p_input['wait'] = self.finished[e]
			p_input['list_of_actions'] = self.list_of_actions_s[e]
			p_input['list_of_actions_pointer'] = self.list_of_actions_pointer_s[e]
			p_input['consecutive_interaction'] = self.consecutive_interaction_s[e]
			p_input['consecutive_target'] = self.target_instance_s[e]
   
			p_input['class_map'] =self.local_map[e, 4:, :, :].cpu().numpy()
			if self.args.visualize or self.args.print_images:
				self.local_map[e, -1, :, :] = 1e-5
				p_input['sem_map_pred'] = self.local_map[e, 4:, :,
					:].argmax(0).cpu().numpy()
				
			if self.first_steps[e]:
				p_input['consecutive_interaction'] = None
				p_input['consecutive_target'] = None
				
		return False #Unfinished

	#def after_step(self, obs, rew, done, infos, goal_success_s, next_step_dict_s):
	def after_step1(self, obs, infos, goal_success_s, next_step_dict_s ):
		#obs, rew, done, infos, goal_success_s, next_step_dict_s = self.envs.plan_act_and_preprocess(self.planner_inputs[0], self.goal_spotted_s[0])
		#pickle.dump(obs, open('temp_pickles/plan_act_obs.p', 'wb')); pickle.dump(infos, open('temp_pickles/plan_act_infos.p', 'wb')); pickle.dump(next_step_dict_s, open('temp_pickles/plan_act_next_step_dict_s.p', 'wb'))
		self.step +=1
		self.obs, self.infos, self.goal_success_s, self.next_step_dict_s = obs, infos, goal_success_s, next_step_dict_s
		self.obs = torch.tensor(np.expand_dims(self.obs, 0)).float().to(self.device); self.infos = (self.infos, ); self.goal_success_s = (self.goal_success_s, ); self.next_step_dict_s = (self.next_step_dict_s,)
		self.goal_success_s = list(self.goal_success_s)
		self.view_angles = []
		for e, p_input in enumerate(self.planner_inputs):
			self.next_step_dict = self.next_step_dict_s[e]
						
			self.view_angle = self.next_step_dict['view_angle']
			
			self.view_angles.append(self.view_angle)
			
			self.num_steps_so_far[e] = self.next_step_dict['steps_taken']
			self.first_steps[e] = False 
			
			self.fails[e] += self.next_step_dict['fails_cur']
			#self.sem_exp.print_log("self.fails[e] is ", self.fails[e])
			#self.sem_exp.print_log("self.next_step_dict['fails_cur'] ", self.next_step_dict['fails_cur'])
			if self.args.leaderboard and self.fails[e] >= self.args.max_fails:
				print("Interact API failed %d times" % self.fails[e] )
				self.task_finish[e] = True

			if not(self.args.no_pickup) and (self.args.map_mask_prop !=1 or self.args.no_pickup_update) and self.next_step_dict['picked_up'] and self.goal_success_s[e]:
				self.do_not_update_cat_s[e] = self.infos[e]['goal_cat_id'] 
			elif not(self.next_step_dict['picked_up']):
				self.do_not_update_cat_s[e] = None
					
			
		self.sem_map_module.set_view_angles(self.view_angles)
			  
		#####################################
		#####################################            
		
		for e, p_input in enumerate(self.planner_inputs):
			if p_input['wait'] ==1 :
				pass
			elif self.next_step_dict_s[e]['keep_consecutive']:
				self.consecutive_interaction_s[e] = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][1]
			else:
				self.consecutive_interaction_s[e], self.target_instance_s[e] = None, None
		self.sem_exp.print_log("AFTER STEP1 self.consecutive_interaction_s[e] set to ", self.consecutive_interaction_s[e])
			
	def after_step1_reset(self):	
		return_actions = []
		for e, p_input in enumerate(self.planner_inputs):

			# if p_input['wait'] ==1  or next_step_dict_s[e]['keep_consecutive']:
   #              pass
   #          else:
   #              consecutive_interaction_s[e], target_instance_s[e] = None, None

			if self.goal_success_s[e]:
				self.sem_exp.print_log("Went into reset! ")
				if self.list_of_actions_pointer_s[e] == len(self.list_of_actions_s[e]) -1:
					self.all_completed[e] = True
				else:
					self.list_of_actions_pointer_s[e] +=1
					self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
					
					self.reset_goal_true_false = [False]* self.num_scenes
					self.reset_goal_true_false[e] = True
					
					
					self.returned, self.target_instance_s[e] = determine_consecutive_interx(self.list_of_actions_s[e], self.list_of_actions_pointer_s[e]-1, self.whether_sliced_s[e])
					if self.returned:
						self.sem_exp.print_log("RESET CONSECUTIVE RETURNED!")
						self.consecutive_interaction_s[e] = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][1]
						self.sem_exp.print_log("self.consecutive_interaction_s[e] set to ", self.consecutive_interaction_s[e])
					self.infos, return_actions = self.envs.reset_goal(self.reset_goal_true_false[e], self.goal_name, self.consecutive_interaction_s[e])
					self.infos = (self.infos, )
					self.goal_spotted_s[e] = False
					self.found_goal[e] = 0
					self.wheres_delete_s[e] = np.zeros((240,240))
						   
		
		time.sleep(self.args.wait_time)
		return return_actions

	def write_log(self):
		for e in range(self.num_scenes):
			self.number_of_this_episode = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
			f = open("results/logs/log_" + self.args.eval_split + "_from_" + str(self.args.from_idx) + "_to_" + str(self.args.to_idx) + "_" + self.dn +".txt" , "a")
			f.write("\n")
			f.write("===================================================\n")
			f.write("episode # is " + str(self.number_of_this_episode) + "\n")
			
			for log in self.next_step_dict_s[e]['logs']:
				f.write(log + "\n")

			if self.all_completed[e]:
				if not(self.finished[e]) and self.args.test:
					f.write("This episode is probably Success!\n")

			if self.num_steps_so_far[e] >= self.args.max_episode_length and not(self.finished[e]):
				f.write("This outputted\n")
			
			print("episode # ", self.number_of_this_episode  , "ended and process number is", e)
			
			
			if self.args.leaderboard and self.args.test:
				actseq = self.next_step_dict_s[e]['actseq']
				self.actseqs.append(actseq)
			f.close()

	def after_step2(self):
		# ------------------------------------------------------------------
		#End episode and log
		for e in range(self.num_scenes):
			self.number_of_this_episode = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
			if self.number_of_this_episode in self.skip_indices:
				self.task_finish[e] = True
		
		for e in range(self.num_scenes):
			if self.all_completed[e]:
				if not(self.finished[e]) and self.args.test:
					print("This episode is probably Success!")
				self.task_finish[e] = True
			
		for e in range(self.num_scenes):
			if self.num_steps_so_far[e] >= self.args.max_episode_length and not(self.finished[e]):
				print("This outputted")
				self.task_finish[e] = True
			
		#Ignore here for now
		# for e in range(self.num_scenes):
		# 	self.number_of_this_episode = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
		# 	if self.task_finish[e] and not(self.finished[e]) and not(self.number_of_this_episode in self.skip_indices): 
		# 		#Add to analyze recs
		# 		analyze_dict = {'task_type': self.actions_dicts[e]['task_type'], 'errs':self.next_step_dict_s[e]['errs'], 'action_pointer':self.list_of_actions_pointer_s[e], 'goal_found':self.goal_spotted_s[e],\
		# 		  'number_of_this_episode': self.number_of_this_episode}
		# 		if not(self.args.test):
		# 			#analyze_dict['success'] = self.envs.evaluate()[1]#[0]
		# 			analyze_dict['success'] = self.all_completed[e]
		# 			print("analyze_dict['success'] is ", analyze_dict['success'])
		# 		else:
		# 			analyze_dict['success'] = self.all_completed[e]
		# 		self.analyze_recs.append(analyze_dict)
		# 		pickle.dump(self.analyze_recs, open("results/analyze_recs/" + self.args.eval_split + "_anaylsis_recs_from_" + str(self.args.from_idx) + "_to_" + str(self.args.to_idx) +  "_" + self.dn +".p", "wb"))
		#################End of File

		##################Beginning of for loop
		if sum(self.finished) == self.args.num_processes:
			# print("all finished")
			# if self.args.leaderboard and self.args.test:
			# 	if self.args.test_seen:
			# 		self.add_str = "seen"
			# 	else:
			# 		self.add_str = "unseen"
			# 	pickle.dump(self.actseqs, open("results/leaderboard/actseqs_test_" + self.add_str + "_" + self.dn + ".p", "wb"))
			
			
			#raise Exception("Finished")
			return True

		self.l_step = self.step % self.args.num_local_steps

	def start_again_if_finished1(self):
		# Reinitialize variables when episode ends
		for e in range(self.num_scenes):
			self.number_of_this_episode = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
			if self.task_finish[e] and not(self.finished[e]) and not(self.number_of_this_episode in self.skip_indices): 
				#Add to analyze recs
				analyze_dict = {'task_type': self.actions_dicts[e]['task_type'], 'errs':self.next_step_dict_s[e]['errs'], 'action_pointer':self.list_of_actions_pointer_s[e], 'goal_found':self.goal_spotted_s[e],\
				  'number_of_this_episode': self.number_of_this_episode, 'edh_instance_true': self.next_step_dict_s[e]['edh_instance_true'], 'total_num_pointers': self.next_step_dict_s[e]['total_num_pointers'],\
				   'list_of_actions': self.next_step_dict_s[e]['list_of_actions']}
				if not(self.args.test):
					#analyze_dict['success'] = self.envs.evaluate()[1]#[0]
					analyze_dict['success'] = self.all_completed[e]
					print("analyze_dict['success'] is ", analyze_dict['success'])
				else:
					analyze_dict['success'] = self.all_completed[e]
				for k, v in self.instance_metrics.items():
					analyze_dict[k] = v
				self.analyze_recs.append(analyze_dict)
				pickle.dump(self.analyze_recs, open("results/analyze_recs/" + self.args.eval_split + "_anaylsis_recs_from_" + str(self.args.from_idx) + "_to_" + str(self.args.to_idx) +  "_" + self.dn +".p", "wb"))


		for e,x in enumerate(self.task_finish):
			if x:
				spl = self.infos[e]['spl']
				success = self.infos[e]['success']
				dist = self.infos[e]['distance_to_goal']
				#spl_per_category[self.infos[e]['goal_name']].append(spl)
				#success_per_category[self.infos[e]['goal_name']].append(success)
				self.traj_number[e] +=1
				#self.wait_env[e] = 1.
				self.init_map_and_pose_for_env(e)
				
				if not(self.finished[e]):
					#load next episode for env
					self.number_of_this_episode = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
					print("steps taken for episode# ",  self.number_of_this_episode-self.num_scenes , " is ", self.next_step_dict_s[e]['steps_taken'])
					self.completed_episodes.append(self.number_of_this_episode)
					pickle.dump(self.completed_episodes, open("results/completed_episodes_" + self.args.eval_split + str(self.args.from_idx) + "_to_" + str(self.args.to_idx) + "_" + self.dn +".p", 'wb'))
					if self.args.leaderboard and self.args.test:
						if self.args.test_seen:
							self.add_str = "seen"
						else:
							self.add_str = "unseen"
						pickle.dump(self.actseqs, open("results/leaderboard/actseqs_test_" + self.add_str + "_" + self.dn + ".p", "wb"))
					self.load = [False] * self.args.num_processes
					self.load[e] = True
					self.do_not_update_cat_s[e] = None
					self.wheres_delete_s[e] = np.zeros((240,240))
					#self.obs, self.infos, self.actions_dicts = self.envs.load_next_scene(self.load[e], edh_instance, start_pose_image)

	#This is after self.envs.load_next_scene has been called
	#def start_again_if_finished2(self,obs, infos, actions_dicts):
	def start_again_if_finished2(self):
		for e,x in enumerate(self.task_finish):
			if x:
				if not(self.finished[e]):
					#self.obs = torch.tensor(np.expand_dims(obs, 0)).float(); self.infos = (infos, ); self.actions_dicts= (actions_dicts, )
					self.obs = torch.tensor(np.expand_dims(self.obs, 0)).float().to(self.device); self.infos = (self.infos, ); self.actions_dicts= (self.actions_dicts, )

					self.view_angles[e] = 45
					self.sem_map_module.set_view_angles(self.view_angles)
					if  self.actions_dicts[e] is None:
						self.finished[e] = True
					else:
						self.second_objects[e] = self.actions_dicts[e]['second_object']
						print("second object is ",self.second_objects[e])
						self.list_of_actions_s[e] = self.actions_dicts[e]['list_of_actions']
						self.task_types[e] = self.actions_dicts[e]['task_type']
						self.whether_sliced_s[e] = self.actions_dicts[e]['sliced']
					
						self.task_finish[e] = False
						self.num_steps_so_far[e] = 0
						self.list_of_actions_pointer_s[e] = 0
						self.goal_spotted_s[e] = False   
						self.found_goal[e] = 0
						self.list_of_actions_pointer_s[e] = 0
						self.first_steps[e] = True
						
						self.all_completed[e] = False
						self.goal_success_s[e] = False
						
						self.obs = torch.tensor(self.obs).to(self.device)
						self.fails[e] = 0
						self.goal_logs[e] = []
						self.goal_cat_before_second_objects[e] = None
						
	def after_step3(self):
		# ------------------------------------------------------------------
		# Semantic Mapping Module
		self.poses = torch.from_numpy(np.asarray(
					[self.infos[env_idx]['sensor_pose'] for env_idx
					 in range(self.num_scenes)])
				).float().to(self.device)


################### Spa relation #####################################

		# for e in range(self.num_scenes):
		# 	self.local_map[e,4:,:,:] = (torch.from_numpy(self.next_step_dict['class_map']).to(self.local_map[e,4:,:,:].device))
#################################################


		# print("Poses is ", self.poses)
		_, self.local_map, _, self.local_pose,self.translated  = self.sem_map_module(self.obs, self.poses, self.local_map, self.local_pose, build_maps = True, no_update = False)

		###### Spa relation  ###################3
		if self.next_step_dict['class_map'].max()>9 :
			for e in range(self.num_scenes):
				self.local_map[e,4:,:,:] *= (torch.from_numpy(self.next_step_dict['class_map']).to(self.local_map[e,4:,:,:].device)/10)
####################################################################3  
  
		locs = self.local_pose.cpu().numpy()
		self.planner_pose_inputs[:, :3] = locs + self.origins
		self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
		for e in range(self.num_scenes):
			r, c = locs[e, 1], locs[e, 0]
			loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
							int(c * 100.0 / self.args.map_resolution)]
			self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

		for e in range(self.num_scenes):
			if not(self.do_not_update_cat_s[e] is None):
				cn = self.do_not_update_cat_s[e] + 4
				self.local_map[e, cn, :, :] = torch.zeros(self.local_map[0, 0, :, :].shape)

		for e in range(self.num_scenes):
			if self.args.delete_from_map_after_move_until_visible and (self.next_step_dict_s[e]['move_until_visible_cycled'] or self.next_step_dict_s[e]['delete_lamp']):
				self.ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
				#Get the label that is closest to the current goal
				cn = self.infos[e]['goal_cat_id'] + 4

				start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.planner_pose_inputs[e]
				gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
				r, c = start_y, start_x
				start = [int(r * 100.0/self.args.map_resolution - gx1),
						 int(c * 100.0/self.args.map_resolution - gy1)]
				map_pred = np.rint(self.local_map[e, 0, :, :].cpu().numpy())
				assert self.local_map[e, 0, :, :].shape[0] == 240
				start = pu.threshold_poses(start, map_pred.shape)

				lm = self.local_map[e, cn, :, :].cpu().numpy()
				lm = (lm>0).astype(int)
				lm = skimage.morphology.binary_dilation(lm, skimage.morphology.disk(4))
				lm = lm.astype(int)
				connected_regions = skimage.morphology.label(lm, connectivity=2)
				unique_labels = [i for i in range(0, np.max(connected_regions)+1)]
				min_dist = 1000000000
				for lab in unique_labels:
					wheres = np.where(connected_regions == lab)
					center = (int(np.mean(wheres[0])), int(np.mean(wheres[1])))
					dist_pose = math.sqrt((start[0] -center[0])**2 + (start[1] -center[1])**2)
					min_dist = min(min_dist, dist_pose)
					if min_dist == dist_pose:
						min_lab = lab

				#Delete that label
				self.wheres_delete_s[e][np.where(connected_regions == min_lab)] = 1

				
		for e in range(self.num_scenes):
			cn = self.infos[e]['goal_cat_id'] + 4
			wheres = np.where(self.wheres_delete_s[e])
			self.local_map[e, cn, :, :][wheres] = 0.0


		# ------------------------------------------------------------------

		# ------------------------------------------------------------------
		# Semantic Policy
		self.newly_goal_set = False
		if self.l_step == self.args.num_local_steps - 1:
			self.newly_goal_set = True
			for e in range(self.num_scenes):
				#if self.wait_env[e] == 1: # New episode
				#	self.wait_env[e] = 0.

				self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
					self.local_map[e]
				self.full_pose[e] = self.local_pose[e] + \
							   torch.from_numpy(self.origins[e]).to(self.device).float()

				locs = self.full_pose[e].cpu().numpy()
				r, c = locs[1], locs[0]
				loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
								int(c * 100.0 / self.args.map_resolution)]

				self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
												  (self.local_w, self.local_h),
												  (self.full_w, self.full_h))

				self.planner_pose_inputs[e, 3:] = self.lmb[e]
				self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
							  self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

				self.local_map[e] = self.full_map[e, :,
							   self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
				self.local_pose[e] = self.full_pose[e] - \
								torch.from_numpy(self.origins[e]).to(self.device).float()


			locs = self.local_pose.cpu().numpy()
			
			for e in range(self.num_scenes):
				self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
				if  self.args.use_sem_policy:
					
					#Just reconst the common map save objects
					map_reconst = torch.zeros((4+len(self.large_objects2idx),240,240))
					map_reconst[:4] = self.local_map[e][:4]
					test_see = {}
					map_reconst[4+self.large_objects2idx['SinkBasin']] = self.local_map[e][4+1]
					test_see[1] = 'SinkBasin'

					start_idx = 2
					for cat, catid in self.large_objects2idx.items():
						if not (cat =='SinkBasin'):
							map_reconst[4+self.large_objects2idx[cat]] = self.local_map[e][4+start_idx]
							test_see[start_idx] = cat
							start_idx +=1

					if self.local_map[e][0][120,120] == 0:
						mask = np.zeros((240,240))
						connected_regions = skimage.morphology.label(1-self.local_map[e][0].cpu().numpy(), connectivity=2)
						connected_lab = connected_regions[120,120]
						mask[np.where(connected_regions==connected_lab)] = 1
						mask[np.where(skimage.morphology.binary_dilation(self.local_map[e][0].cpu().numpy(), skimage.morphology.square(4)))] = 1
					else:
						dilated = skimage.morphology.binary_dilation(self.local_map[e][0].cpu().numpy(), skimage.morphology.square(4))
						mask = skimage.morphology.convex_hull_image(dilated).astype(float)
					self.mask_grid = into_grid(torch.tensor(mask), 8).cpu() 
					self.where_ones = len(torch.where(self.mask_grid)[0])
					self.mask_grid = self.mask_grid.repeat(73,1).view(73, -1).numpy()

					if  self.goal_name in self.all_objects2idx and self.next_step_dict_s[e]['steps_taken'] >= 30:

						pred_probs = self.Unet_model(map_reconst.unsqueeze(0).to(self.device))
						pred_probs = pred_probs.view(73, -1)
						pred_probs = self.softmax(pred_probs).cpu().numpy()

						pred_probs = (1-self.args.explore_prob) * pred_probs + self.args.explore_prob * self.mask_grid * 1/ float(self.where_ones)

						#Now sample from pred_probs according to goal idx
						self.goal_name = self.list_of_actions_s[e][self.list_of_actions_pointer_s[e]][0]
						if self.goal_name =='FloorLamp':
							pred_probs = pred_probs[self.all_objects2idx[self.goal_name]] + pred_probs[self.all_objects2idx['DeskLamp']]
							pred_probs = pred_probs/2.0
						else:
							pred_probs = pred_probs[self.all_objects2idx[self.goal_name]]


					else:
						pred_probs =  self.mask_grid[0] * 1/ float(self.where_ones)

					

					
					if self.args.explore_prob==1.0:
						mask_wheres = np.where(mask.astype(float))
						np.random.seed(self.next_step_dict_s[e]['steps_taken'])
						s_i= np.random.choice(len(mask_wheres[0]))
						x_240, y_240 = mask_wheres[0][s_i], mask_wheres[1][s_i]

					else:
						#Now sample one index
						np.random.seed(self.next_step_dict_s[e]['steps_taken'])
						pred_probs = pred_probs.astype('float64')
						pred_probs = pred_probs.reshape(64)
						pred_probs = pred_probs/ np.sum(pred_probs)

						chosen_cell = np.random.multinomial(1, pred_probs.tolist())
						chosen_cell = np.where(chosen_cell)[0][0]                        
						chosen_cell_x = int(chosen_cell/8)
						chosen_cell_y = chosen_cell %8
						
						
						#Sample among this mask
						mask_new = np.zeros((240,240))
						mask_new[chosen_cell_x*30:chosen_cell_x*30+30, chosen_cell_y*30: chosen_cell_y*30 + 30] = 1 
						mask_new = mask_new * mask
						if np.sum(mask_new) == 0:
							np.random.seed(self.next_step_dict_s[e]['steps_taken'])
							chosen_i = np.random.choice(len(np.where(mask)[0]))
							x_240 = np.where(mask)[0][chosen_i]
							y_240 = np.where(mask)[1][chosen_i]

						else:
							np.random.seed(self.next_step_dict_s[e]['steps_taken'])
							chosen_i = np.random.choice(len(np.where(mask_new)[0]))
							x_240 = np.where(mask_new)[0][chosen_i]
							y_240 = np.where(mask_new)[1][chosen_i]
						


					self.global_goals[e] = [x_240, y_240]
					self.test_goals= np.zeros((240,240))
					self.test_goals[x_240,y_240]=1

							

		# ------------------------------------------------------------------
	
	
		# ------------------------------------------------------------------
		# Take action and get next observation
		self.found_goal = [0 for _ in range(self.num_scenes)]
		self.goal_maps = [np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)]

		for e in range(self.num_scenes):
			self.goal_maps[e][self.global_goals[e][0], self.global_goals[e][1]] = 1


		for e in range(self.num_scenes):
			ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
			cn = self.infos[e]['goal_cat_id'] + 4
			self.prev_cns[e] = cn
			cur_goal_sliced = self.next_step_dict_s[e]['current_goal_sliced']

			if self.local_map[e, cn, :, :].sum() != 0.: 
				ep_num = self.args.from_idx + self.traj_number[e] * self.num_scenes + e
				cat_semantic_map = self.local_map[e, cn, :, :].cpu().numpy()
				cat_semantic_scores = cat_semantic_map 

				cat_semantic_scores[cat_semantic_scores > 0] = 1.
				wheres = np.where(self.wheres_delete_s[e])
				cat_semantic_scores[wheres] = 0
				if np.sum(cat_semantic_scores) !=0:
					self.goal_maps[e] = cat_semantic_scores

				if np.sum(cat_semantic_scores) !=0:
					self.found_goal[e] = 1
					self.goal_spotted_s[e] = True
				else:
					if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
						self.found_goal[e] = 0
						self.goal_spotted_s[e] = False
			else:
				if self.args.delete_from_map_after_move_until_visible or self.args.delete_pick2:
					self.found_goal[e] = 0
					self.goal_spotted_s[e] = False



		self.planner_inputs = [{} for e in range(self.num_scenes)]
		for e, p_input in enumerate(self.planner_inputs):
			p_input['newly_goal_set'] =self.newly_goal_set
			p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
			p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
			p_input['pose_pred'] = self.planner_pose_inputs[e]
			p_input['goal'] = self.goal_maps[e] 
			p_input['new_goal'] = self.l_step == self.args.num_local_steps - 1
			p_input['found_goal'] = self.found_goal[e]
			p_input['wait'] = self.finished[e]
			p_input['list_of_actions'] = self.list_of_actions_s[e]
			p_input['list_of_actions_pointer'] = self.list_of_actions_pointer_s[e]
			p_input['consecutive_interaction'] = self.consecutive_interaction_s[e]
			p_input['consecutive_target'] = self.target_instance_s[e]
   
			p_input['class_map'] =self.local_map[e, 4:, :, :].cpu().numpy()
			if self.args.visualize or self.args.print_images:
				self.local_map[e, -1, :, :] = 1e-5
				p_input['sem_map_pred'] = self.local_map[e, 4:, :,
					:].argmax(0).cpu().numpy()
				
			if self.first_steps[e]:
				p_input['consecutive_interaction'] = None
				p_input['consecutive_target'] = None
						
		return False #Unfinished
