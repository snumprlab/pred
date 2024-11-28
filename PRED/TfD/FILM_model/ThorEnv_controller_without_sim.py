import os
import sys

import cv2
import copy
import alfred_utils.gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from alfred_utils.env.tasks import get_task
from ai2thor.controller import Controller
import alfred_utils.gen.utils.image_util as image_util
from alfred_utils.gen.utils import game_util
from alfred_utils.gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj
import torch
from torch.autograd import Variable
from models.depth.alfred_perception_models import AlfredSegmentationAndDepthModel

from arguments import get_args

import quaternion
import gym

import envs.utils.pose as pu

import matplotlib

if sys.platform == 'darwin':
	matplotlib.use("tkagg")
else:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pickle
from types import SimpleNamespace
import skimage.morphology



DEFAULT_RENDER_SETTINGS = {'renderImage': True,
						   'renderDepthImage': True,
						   'renderClassImage': False,
						   'renderObjectImage': True,
						   }


class ThorEnv_ControllerWithoutSim:
	def __init__(self, args, rank):
		self.steps_taken = 0
		
		self.task = None

		# internal states
		self.cleaned_objects = set()
		self.cooled_objects = set()
		self.heated_objects = set()

		#initializations compatible with object_goal_env
		if args.visualize:
			plt.ion()
		if args.print_images or args.visualize:
			if args.visualize == 1:
				widths = [4, 4, 3]
				total_w = 11
			elif args.visualize == 2:
				widths = [4, 3, 3]
				total_w = 10
			else:
				widths = [4, 3]
			self.figure, self.ax = plt.subplots(1, len(widths), figsize=(6*16./9., 6),
												gridspec_kw={'width_ratios': widths},
												facecolor="whitesmoke",
												num="Thread {}".format(rank))
		
		self.args = args 
		self.rank = rank
		
		self.episode_no = 0
		self.last_scene_path = None
		self.info = {}
		self.info['distance_to_goal'] = None
		self.info['spl'] = None
		self.info['success'] = None
		
		self.view_angle = 0
		self.consecutive_steps = False
		self.final_sidestep = False
		
		self.actions = []
		self.max_depth = 5
		self.depth = None

		if args.use_learned_depth:
			self.depth_gpu =   torch.device("cuda" if torch.cuda.is_available() else "cpu")
			if not(args.valts_depth):
				bts_args = SimpleNamespace(model_name='bts_nyu_v2_pytorch_densenet161' ,
				encoder='densenet161_bts',
				dataset='alfred',
				input_height=300,
				input_width=300,
				max_depth=5,
				mode = 'test',
				device = self.depth_gpu,
				set_view_angle=False,
				load_encoder=False,
				load_decoder=False,
				bts_size=512)

				if self.args.depth_angle or self.args.cpp:
					bts_args.set_view_angle = True
					bts_args.load_encoder = True

				self.depth_pred_model = BtsModel(params=bts_args).to(device=self.depth_gpu)
				print("depth initialized")

				if args.cuda:
					ckpt_path = 'models/depth/depth_models/' + args.depth_checkpoint_path 
				else:
					ckpt_path = 'models/depth/depth_models/' + args.depth_checkpoint_path 
				checkpoint = torch.load(ckpt_path, map_location=self.depth_gpu)['model']

				new_checkpoint = OrderedDict()
				for k, v in checkpoint.items():
					name = k[7:] # remove `module.`
					new_checkpoint[name] = v
				del checkpoint
				# load params
				self.depth_pred_model.load_state_dict(new_checkpoint)
				self.depth_pred_model.eval()
				self.depth_pred_model.to(device=self.depth_gpu)

			#Use Valts depth
			else:

				#model_path ='valts/model-2000-best_silog_10.13741' #45 degrees only model
				#model_path = 'height180_valts60/model-31000-best_silog_27.80921'
				model_path = 'may16_height180_valts60/model-16000-best_silog_18.00671'
				if self.args.depth_model_old:
					model_path ='valts/model-34000-best_silog_16.80614'

				elif self.args.depth_model_45_only:
					model_path = 'valts/model-500-best_d3_0.98919'

				self.depth_pred_model = AlfredSegmentationAndDepthModel()

				state_dict = torch.load(os.path.join(os.environ["FILM_model_dir"], 'models/depth/depth_models/' +model_path), map_location=self.depth_gpu)['model']

				new_checkpoint = OrderedDict()
				for k, v in state_dict.items():
					name = k[7:] # remove `module.`
					new_checkpoint[name] = v
				
				state_dict = new_checkpoint
				del new_checkpoint

				self.depth_pred_model.load_state_dict(state_dict)
				self.depth_pred_model.eval()
				self.depth_pred_model.to(device=self.depth_gpu)

				if self.args.separate_depth_for_straight:
					#model_path = 'valts0/model-102500-best_silog_17.00430'
					#model_path = 'height180_valts0/model-12500-best_silog_37.60476'
					model_path = 'may16_height180_valts0/model-49000-best_d2_0.89905'
					

					self.depth_pred_model_0 = AlfredSegmentationAndDepthModel()
					state_dict = torch.load( os.path.join(os.environ["FILM_model_dir"], 'models/depth/depth_models/' +model_path), map_location=self.depth_gpu)['model']

					new_checkpoint = OrderedDict()
					for k, v in state_dict.items():
						name = k[7:] # remove `module.`
						new_checkpoint[name] = v
					
					state_dict = new_checkpoint
					del new_checkpoint

					self.depth_pred_model_0.load_state_dict(state_dict)
					self.depth_pred_model_0.eval()
					self.depth_pred_model_0.to(device=self.depth_gpu)
					
		
		print("ThorEnv started.")

	def setup_scene(self, starting_hor):
		'''
		intialize the scene and agent from the task info
		'''
		# scene setup
		self.view_angle = 60
		#scene_num = traj_data['scene']['scene_num']
		#object_poses = traj_data['scene']['object_poses']
		#dirty_and_empty = traj_data['scene']['dirty_and_empty']
		#object_toggles = traj_data['scene']['object_toggles']
		self.steps_taken = 0
		self.errs = []
		self.actions = [dict(action="LookDown_15", forceAction=True)]
		if not(self.args.approx_horizon):
			#self.camera_horizon =30
			self.camera_horizon =starting_hor
		else:
			#self.camera_horizon = 30
			self.camera_horizon =starting_hor
		#scene_name = 'FloorPlan%d' % scene_num
		state, info = self.reset()
		#self.restore_scene(object_poses, object_toggles, dirty_and_empty)
		#if not(self.args.test):
		#    self.set_task(traj_data, task_type, args, task_type_optional=task_type)

		# initialize to start position
		#init_dict = dict(traj_data['scene']['init_action'])
		#obs, _, _, info = self.step(init_dict)
		#if not(self.args.approx_horizon):
		#    self.camera_horizon =self.event.metadata['agent']['cameraHorizon']
		#else:
		#    self.camera_horizon = 30
		#obs, _, _, info = self.step(dict(action='LookDown_15',
		#                      forceAction=True))
		#objects = self.event.metadata["objects"]
		#tables = []
		
		# if not(r_idx is None):
		#     print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

		# # setup task for reward
		# if not(self.args.test):
		#     self.set_task(traj_data, task_type, args, reward_type=reward_type, task_type_optional=task_type)
		# self.info = info
		
		# return obs, info
		return state, info

	def print_log(self, *statements):
		statements = [str(s) for s in statements]
		statements = ['step #: ', str(self.steps_taken) , ","] + statements
		joined = ' '.join(statements)
		#print(joined)
		self.logs.append(joined)

	def reset(self):
		'''
		reset scene and task states
		'''
		print("Resetting ThorEnv")

		# if type(scene_name_or_num) == str:
		#     scene_name = scene_name_or_num
		# else:
		#     scene_name = 'FloorPlan%d' % scene_name_or_num

		self.accumulated_pose = np.array([0.0,0.0,0.0])
		# super().reset(scene_name)
		# event = super().step(dict(
		#     action='Initialize',
		#     gridSize=grid_size,
		#     cameraY=camera_y,
		#     renderImage=render_image,
		#     renderDepthImage=render_depth_image,
		#     renderClassImage=render_class_image,
		#     renderObjectImage=render_object_image,
		#     visibility_distance=visibility_distance,
		#     makeAgentsVisible=False,
		# ))
		#self.event = event

		# reset task if specified
		#if self.task is not None:
		#    self.task.reset()

		# clear object state changes
		#self.reset_states()
		
		self.timestep = 0
		self.stopped = False
		self.path_length = 1e-5
		
		self.info['time'] = self.timestep
		self.info['sensor_pose'] = [0., 0., 0.]
		
		if not(self.args.approx_pose):
			self.last_sim_location = self.get_sim_location()
		self.o = 0.0
		self.o_behind = 0.0
		
		# if return_event:
		#     return event
		# else:
		#     rgb = torch.tensor(event.frame.copy()).numpy() #shape (h, w, 3)
		#     depth = torch.tensor(event.depth_frame.copy()).numpy() #shape (h, w)
		#     depth /= 1000.0
		#     depth = np.expand_dims(depth, 2)
			
		#     state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
			
		#     return state, self.info

		state = self.get_obs()
		state[:,:,:] = 0
		return state, self.info

	#Get obs for internal step
	def get_obs(self):
		#get obs from self.img
		rgb = self.img  #shape (h, w, 3)
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

		#depth = torch.tensor(self.event.depth_frame.copy()).numpy() #shape (h, w) #TODO: Change this!
		if not(self.depth is None):
			depth = torch.tensor(self.depth.copy()).numpy()
		else:	
			depth = torch.zeros((self.img.shape[0], self.img.shape[1])).numpy() #shape (h, w) 
			depth /= 1000.0 #in meters

		depth = np.expand_dims(depth, 2)
		
		state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
				
		return state #This is obs 

	def thor_step_internal_right_after_sim_step(self):
		#Horizon
		#if 'LookUp' in prev_action and self.success_for_look(action['action']):
		#	self.camera_horizon += -angle
		#elif 'LookDown' in prev_action and self.success_for_look(action['action']):
		#	self.camera_horizon += angle
		#print("prev action is ", self.prev_action)
		if 'Look' in self.prev_action:
			angle = float(self.prev_action.split("_")[1])
			#cv2.imwrite('prev_rgb.png', self.prev_rgb)
			#cv2.imwrite('cur_rgb.png', self.img)
			#print("Success for look is ", self.success_for_look(self.prev_action))
			if 'LookUp' in self.prev_action and self._get_approximate_success_for_pose(self.prev_rgb, self.img, self.prev_action):
				self.camera_horizon += -angle
			elif 'LookDown' in self.prev_action and self._get_approximate_success_for_pose(self.prev_rgb, self.img, self.prev_action):
				# print("Went through!")
				self.camera_horizon += angle
			
		# Get pose change
		if not(self.args.approx_pose):
			if self.consecutive_steps == False:
				dx, dy, do = self.get_pose_change()
				self.info['sensor_pose'] = [dx, dy, do]
				self.path_length += pu.get_l2_distance(0, dx, 0, dy)
			else:
				if self.final_sidestep:
					dx, dy, do = self.get_pose_change()
					self.info['sensor_pose'] = [dx, dy, do]
					self.path_length += pu.get_l2_distance(0, dx, 0, dy)
				else:
					pass
		else:
			if self.args.approx_last_action_success:
				#whether_success = self.success_for_look(action['action'])
				#self.print_log("Approx last action success ; prev action was ", self.prev_action)
				whether_success = self._get_approximate_success_for_pose(self.prev_rgb, self.img, self.prev_action)#whether_success = self.success_for_look(self.prev_action)
				#self.print_log("whether success is ", whether_success)
			else:
				whether_success  = self.event.metadata['lastActionSuccess']
			if self.consecutive_steps:
				#last_action = action['action']
				last_action = self.prev_action
				dx, dy, do = self.get_pose_change_approx_relative(last_action, whether_success)
				self.accumulated_pose += np.array([dx, dy, do])
				if self.accumulated_pose[2]  >= np.pi -1e-1:
					self.accumulated_pose[2]  -= 2 * np.pi
				if self.final_sidestep:
					self.info['sensor_pose'] = copy.deepcopy(self.accumulated_pose).tolist()
					self.path_length += pu.get_l2_distance(0, dx, 0, dy)
					self.accumulated_pose = np.array([0.0,0.0,0.0])
					self.o = 0.0
			else:
				#last_action = action['action']
				last_action = self.prev_action
				dx, dy, do = self.get_pose_change_approx(last_action, whether_success)
				self.info['sensor_pose'] = [dx, dy, do]
				self.path_length += pu.get_l2_distance(0, dx, 0, dy)

		#done = self.get_done()
		
		# if done:
		# 	spl, success, dist = 0,0,0
		# 	self.info['distance_to_goal'] = dist
		# 	self.info['spl'] = spl
		# 	self.info['success'] = success
		
		
		self.timestep += 1
		self.info['time'] = self.timestep
		
		state = self.get_obs()
		
		self.steps_taken +=1
		return state, self.info


	#All the step functions of thor_env_code such as to_thor_api_exec, va_interact_new, step
	#def thor_step_internal_right_before_sim_step(self, planner_inputs):
	def thor_step_internal_right_before_sim_step(self):
		#Final thing before calling step in the simulator
		#self.action_received = last_predicted_action
		self.action_received = self.prev_action
		self.prev_rgb = copy.deepcopy(self.img)

	def _get_max_area(self, frame1, frame2, mask=None):
		wheres = np.where(frame1 != frame2)
		wheres_ar = np.zeros(frame1.shape)
		wheres_ar[wheres] = 1
		if not(mask is None):
			#Set areas outside mask as 0
			wheres_non_mask = np.where(mask==0)
			wheres_ar[wheres_non_mask] = 0
		wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
		connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
		unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
		max_area = -1
		for lab in unique_labels:
			wheres_lab = np.where(connected_regions == lab)
			max_area = max(len(wheres_lab[0]), max_area)
		return max_area

	def _get_approximate_success_for_pose(self, prev_rgb, frame, action):
		success = False
		prev_rgb = copy.deepcopy(prev_rgb)
		#prev_rgb[250:, :] = 0
		frame = copy.deepcopy(frame)
		#frame[250:, :] = 0
		if (action in ['MoveAhead_25', "RotateLeft_90", "RotateRight_90", "LookDown_30", "LookUp_30"]): #Max area of just faucet being on is 448 when agent is close
			max_area = self._get_max_area(prev_rgb, frame)
			self.print_log("Action is ", action)
			self.print_log("Max area is ", max_area)
			if max_area>9000:
				success = True
			self.print_log("Approx Success is ", success)
		return success

	###########################
	### Pose
	###########################
	def get_pose_change_approx(self, last_action, whether_success):

		if not(whether_success):
			self.print_log("Pose change approx is ", 0.0, 0.0, 0.0)
			return 0.0, 0.0, 0.0
		else:
			if "MoveAhead" in last_action:
				dx, dy, do = 0.25, 0.0, 0.0
			elif "RotateLeft" in last_action:
				dx, dy = 0.0, 0.0
				do = np.pi/2
			elif "RotateRight" in last_action:
				dx, dy = 0.0, 0.0
				do = -np.pi/2
			else:
				dx, dy, do = 0.0, 0.0, 0.0

			return dx, dy, do 

	def get_pose_change_approx_relative(self, last_action, whether_success):
		if not(whether_success):
			return 0.0, 0.0, 0.0
		else:
			if "MoveAhead" in last_action:
				do = 0.0
				if abs(self.o + 2*np.pi) <=1e-1 or abs(self.o) <=1e-1 or abs(self.o - 2*np.pi) <=1e-1: #o is 0
					dx = 0.25
					dy = 0.0
				elif abs(self.o + 2*np.pi - np.pi/2) <=1e-1 or abs(self.o - np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - np.pi/2) <=1e-1:
					dx = 0.0
					dy = 0.25
				elif abs(self.o + 2*np.pi - np.pi) <=1e-1 or abs(self.o - np.pi) <=1e-1 or abs(self.o - 2*np.pi - np.pi) <=1e-1:
					dx = -0.25
					dy = 0.0
				elif abs(self.o + 2*np.pi - 3*np.pi/2) <=1e-1 or abs(self.o - 3*np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - 3*np.pi/2) <=1e-1:
					dx = 0.0
					dy = -0.25
				else:
					raise Exception("angle did not fall in anywhere")
			elif "RotateLeft" in last_action:
				dx, dy = 0.0, 0.0
				do = np.pi/2
			elif "RotateRight" in last_action:
				dx, dy = 0.0, 0.0
				do = -np.pi/2
			else:
				dx, dy, do = 0.0, 0.0, 0.0

		self.o = self.o + do
		if self.o  >= np.pi- 1e-1:
			self.o  -= 2 * np.pi

		return dx, dy, do 

	###################
	## Camera Horizon
	####################
	# def success_for_look(self, action):
	# 	#wheres = np.where(self.prev_rgb != self.event.frame)
	# 	wheres = np.where(self.prev_rgb != self.img)
	# 	wheres_ar = np.zeros(self.prev_rgb.shape)
	# 	wheres_ar[wheres] = 1
	# 	wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
	# 	connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
	# 	unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
	# 	max_area = -1
	# 	for lab in unique_labels:
	# 		wheres_lab = np.where(connected_regions == lab)
	# 		max_area = max(len(wheres_lab[0]), max_area)
	# 	if action in ['OpenObject', 'CloseObject'] and max_area > 500:
	# 		success = True
	# 	elif max_area > 100:
	# 		success = True
	# 	else:
	# 		success = False
	# 	return success

	def get_pose_change(self):
		curr_sim_pose = self.get_sim_location()
		dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
		self.last_sim_location = curr_sim_pose
		return dx, dy, do

	def get_sim_location(self):
		y = -self.event.metadata['agent']['position']['x']
		x = self.event.metadata['agent']['position']['z'] 
		o = np.deg2rad(-self.event.metadata['agent']['rotation']['y'])
		if o > np.pi:
			o -= 2 * np.pi
		return x, y, o

	###############
	## noop -> Do in the main real
	###############

	
