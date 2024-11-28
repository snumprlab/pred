#Class that will replace thor_env_code and sem_exp_thor now
import math
import os, sys

import matplotlib
if sys.platform == 'darwin':
	matplotlib.use("tkagg")
else:
	matplotlib.use('Agg')

import pickle, json
import copy
import string

import torch
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import skimage.morphology
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from envs.utils.fmm_planner import FMMPlanner
import envs.utils.pose as pu
import alfred_utils.gen.constants as constants
from ThorEnv_controller_without_sim import ThorEnv_ControllerWithoutSim
#from models.instructions_processed_LP.ALFRED_task_helper import get_list_of_highlevel_actions, determine_consecutive_interx, get_arguments, get_arguments_test, read_test_dict
#from models.instructions_processed_LP.ALFRED_task_helper_teach import get_task_type_and_params, list_of_parameters, get_list_of_highlevel_actions #import get_list_of_highlevel_actions, determine_consecutive_interx, get_arguments, get_arguments_test, read_test_dict
from models.segmentation.segmentation_helper import SemgnetationHelper
from models.segmentation.segmentation_helper import SemgnetationHelper
#from models.depth.depth_helper import DepthHelper 
import utils.control_helper as  CH

from models.instructions_processed_LP.object_state import ObjectState
from models.instructions_processed_LP.get_arguments_from_bert import GetArguments


class SemExp_ControllerWithoutSim(ThorEnv_ControllerWithoutSim): 
	#Took init from sem_exp_thor
	def __init__(self, args, rank):
		self.fails_cur = 0

		self.args = args


		super().__init__(args, rank)

		# initialize transform for RGB observations
		self.res = transforms.Compose([transforms.ToPILImage(),
					transforms.Resize((args.frame_height, args.frame_width),
									  interpolation = Image.NEAREST)])
		

		# initializations for planning:
		self.selem = skimage.morphology.square(self.args.obstacle_selem)
		if self.args.disk_selem:
			self.selem = skimage.morphology.disk(self.args.obstacle_selem)
		self.flattened = pickle.load(open("src/teach/inference/FILM_refactor_april1/miscellaneous/flattened.p", "rb"))
		
		self.last_three_sidesteps = [None]*3
		self.picked_up = False
		self.picked_up_cat = None
		self.picked_up_mask = None
		
		self.transfer_cat = {'ButterKnife': 'Knife', "Knife":"ButterKnife"}
		
		#self.scene_names = scene_names
		self.scene_pointer = -1
		
		self.obs = None
		self.steps = 0
		
		self.test_dict = None
		self.lookDownUpLeft_count = 0
		self.action_5_count = 0
		self.goal_visualize = None
		
		#self.test_dict = read_test_dict(self.args.test, self.args.appended, 'unseen' in self.args.eval_split)
		self.test_dict =  None

		#Segmentation
		self.seg = SemgnetationHelper(self)

		if self.args.use_bert:
			self.ArgPredictor= GetArguments()


	def find_starting_horizon(self):
		actions = [ei['action_name']for ei in self.edh_instance['driver_action_history']]
		#count the number of lookups and lookdowns
		num_look_downs = sum([a=="Look Down" for a in actions])
		num_look_ups = sum([a=="Look Up" for a in actions])
		starting_hor = 30 + 30 *(num_look_downs - num_look_ups)
		return starting_hor


	def load_initial_scene(self, edh_instance, image):
		self.scene_pointer += 1
		self.img = image
		self.side_step_order = 0
		self.PREV_side_step_order = 0
		self.rotate_before_side_step_count = 0
		self.fails_cur = 0
		self.put_rgb_mask = None
		self.pointer = 0
		self.interaction_mask = None

		self.execute_interaction = False
		self.interaction_mask = None

		self.prev_rgb = None
		self.prev_depth = None
		self.prev_seg = None

		self.steps_taken = 0
		self.goal_name = None
		self.steps = 0
		#self.last_err = ""
		self.prev_wall_goal = None
		self.dilation_deg = 0
		self.prev_number_action = None
		self.move_until_visible_order = 0
		self.consecutive_steps = False
		self.repeating_sidestep = 0
		self.cat_equate_dict = {} #map "key" category to "value" category
		self.rotate_aftersidestep = None
		self.errs = []
		self.logs = []

		
		exclude = set(string.punctuation)

		self.broken_grid = []
		self.where_block = []
		self.remove_connections_to = None
		
		self.lookDownUpLeft_count = 0
		self.action_5_count = 0
		
		self.last_three_sidesteps = [None]*3
		self.picked_up = False
		self.picked_up_cat = None
		self.picked_up_mask = None

		episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank        
		self.episode_no = episode_no

		#try:
		#traj_data = self.load_traj(self.scene_names[self.scene_pointer]); r_idx = self.scene_names[self.scene_pointer]['repeat_idx']
		#self.traj_data = traj_data = self.load_traj({'repeat_idx': 0, 'task': 'trial_T20190906_190741_530882'})
		traj_data = edh_instance
		self.traj_data = traj_data
		self.r_idx = 0; r_idx = self.r_idx
		#print("Traj data is ", self.traj_data)
		
		self.picture_folder_name = "pictures/" + self.args.eval_split + "/"+ self.args.dn + "/" + str(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank) + "/"
		if self.args.save_pictures and not (self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank in self.args.skip_indices):
			os.makedirs(self.picture_folder_name)
			os.makedirs(self.picture_folder_name + "/fmm_dist")
			os.makedirs(self.picture_folder_name + "/Sem")
			os.makedirs(self.picture_folder_name + "/Rgb")
			os.makedirs(self.picture_folder_name + "/SinkMasks")
			os.makedirs(self.picture_folder_name + "/Sem_Map")
			os.makedirs(self.picture_folder_name + "/edh_dir")
			#os.makedirs(self.picture_folder_name + "/gt_depth")
		

		###################################################################################################################################################
		###LANGUAGE PLANNING######################################################################################################################################
		self.edh_instance = edh_instance

		if self.args.use_bert:
			task_type, obj_count, obj_target, parent_target = self.ArgPredictor.get_pred(edh_instance)
			if (task_type in self.ArgPredictor.Types_we_want_obj) and obj_target == None:
				return None, None, None, None
			if (task_type in self.ArgPredictor.Types_we_want_parent) and parent_target == None:
				return None, None, None, None

		else:
			from models.instructions_processed_LP.object_state_helper import _get_task_type_and_params
			tmp = _get_task_type_and_params(self.edh_instance)
			task_type, obj_count, obj_target, parent_target = tmp["task_type"], tmp["obj_count"], tmp["obj_target"], tmp["parent_target"]

		real_obj_target = copy.deepcopy(obj_target); real_parent_target = copy.deepcopy(parent_target)
		if obj_target in ['Tableware', 'Drinkware', 'Silverware', 'SmallHandheldObjects']:
			if obj_target =='Tableware':
				obj_target = "Plate"
			elif obj_target == 'Drinkware':
				obj_target = "Mug"
			elif obj_target == 'SmallHandheldObject':
				obj_target = "RemoteControl"
			elif obj_target == "Silverware":
				obj_target = "Fork"

		if parent_target in ["Tables"]:
			parent_target = "DiningTable"


		prev_actions = edh_instance['driver_action_history']
		history_subgoals = edh_instance['history_subgoals'] 
		obj_state = ObjectState(edh_instance['dialog_history'], prev_actions, history_subgoals, task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target)
		sliced = obj_state.slice_needs_to_happen
		#sliced= False
		#list_of_actions, categories_in_inst, second_object, caution_pointers = get_list_of_highlevel_actions(task_type, obj_count, obj_target, parent_target, sliced)
		#####################
		list_of_actions = obj_state.future_list_of_highlevel_actions 
		categories_in_inst = obj_state.categories_in_inst
		#Breadsliced and Houseplant that were forcefully added
		# if not('HousePlant' in categories_in_inst ):
		# 	categories_in_inst.append('HousePlant')
		# if 'Bread' in categories_in_inst:
		# 	for i, c in enumerate(categories_in_inst):
		# 		if categories_in_inst[i] == 'Bread':
		# 			categories_in_inst[i] = 'BreadSliced'
		# 	for i, tup in enumerate(list_of_actions):
		# 		if tup[0] == 'Bread':
		# 			list_of_actions[i] = ('BreadSliced', tup[1])
		# if not('BreadSliced' in categories_in_inst ):
		# 	categories_in_inst.append('BreadSliced')
		second_object = obj_state.second_object
		caution_pointers = obj_state.caution_pointers
		if self.args.add_faucet_toggleoff:
			if task_type in ['Coffee', 'Clean All X']:
				list_of_actions = [('Faucet', 'ToggleObjectOff')] + list_of_actions
				if not ('Faucet' in categories_in_inst):
					categories_in_inst = ['Faucet'] + categories_in_inst
				caution_pointers = [i+1 for i in caution_pointers]
				#Ignore second object for now
		self.first_goal_sliced_state = False
		for o in obj_state.init_obj_state_dict:
			if obj_state.init_obj_state_dict[o]["Sliced"]:
				self.first_goal_sliced_state = True
		#Set to false if holding knife
		#if "Knife" in obj_state.init_obj_state_dict and obj_state.init_obj_state_dict["Knife"]['Pickedup']:



		#list_of_actions = [('Knife', 'PickupObject'), ('CounterTop', 'PutObject')]; sliced=True
		#categories_in_inst = ['Knife', 'CounterTop', 'SaltShaker', 'PepperShaker', 'Fridge', 'StoveBurner', 'SinkBasin', 'Toaster', 'Lettuce', 'Mug', 'CoffeeTable']
		#list_of_actions = [("Dresser", "PutObject"), ('Fridge', 'OpenObject')]; sliced=True
		#categories_in_inst =["Cabinet", "Drawer", "Desk", "Sofa", 'TVStand', 'Dresser', "TissueBox", "Fridge"]
		#categories_in_inst =['Dresser', "TissueBox", "Fridge"]
		#caution_pointers = [1]

		#Save these too
		if self.args.save_pictures:
			pickle.dump(edh_instance, open(self.picture_folder_name +"/edh_dir/" + str(self.steps_taken) +'_edh_instance.p', 'wb')) 
			pickle.dump(obj_state, open(self.picture_folder_name +"/edh_dir/" + str(self.steps_taken) +'_obj_state.p', 'wb')) 

		self.print_log("Task type is ", task_type)
		self.print_log("Unmatched states are ", obj_state.unmatched_states)
		self.print_log("List of actions ", list_of_actions); print("List of actions ", list_of_actions)
		self.print_log("categories_in_inst ", categories_in_inst); print("categories_in_inst ", categories_in_inst)
		self.print_log("Expected state changes: ",  edh_instance['state_changes'])#; print("Expected state changes: ",  edh_instance['state_changes'])
		self.print_log("Edh ID: ",  edh_instance['instance_id']); print("Edh ID: ",  edh_instance['instance_id'])


		####FOR BERT, TAKE CARE OF NONE CLASSES
		if self.args.use_bert:
			if (task_type in self.ArgPredictor.Types_we_want_obj) and obj_target == None:
				list_of_actions  = []
			if (task_type in self.ArgPredictor.Types_we_want_parent) and parent_target == None:
				list_of_actions  = []

		if list_of_actions == []:
			return None, None, None, None

		self.sliced = sliced
		self.caution_pointers = caution_pointers
		if self.args.no_caution_pointers:
			self.caution_pointers = []
		self.print_log("list of actions is ", list_of_actions)
		self.task_type = task_type
		self.second_object = second_object
		###################################################################################################################################################
		 
		self.reset_total_cat_new(categories_in_inst)
		#Now we will do set up scene after the three look down's
		#print("SETUP SCENE START")
		#obs, info = self.setup_scene_step0() 
		starting_hor = self.find_starting_horizon()
		obs, info = super().setup_scene(starting_hor)
		#print("SETUP SCENE END")
		goal_name = list_of_actions[0][0]
		print("RESET GOAL START")
		info, return_actions = self.reset_goal(True, goal_name, None)
		print("RESET GOAL END")

		#if task_type == 'look_at_obj_in_light':
		#	self.total_cat2idx['DeskLamp'] = self.total_cat2idx['FloorLamp'] 
		#	self.cat_equate_dict['DeskLamp'] = 'FloorLamp' #if DeskLamp is found, consider it as FloorLamp


		if task_type == "Water Plant":
			for obj in obj_state.FILLABLE_OBJS:
				if obj_state.intermediate_obj in self.total_cat2idx:
					self.total_cat2idx[obj] = self.total_cat2idx[obj_state.intermediate_obj]
					self.cat_equate_dict[obj] = obj_state.intermediate_obj

		if real_obj_target in ['Tableware', 'Drinkware', 'Silverware', 'SmallHandheldObjects']:
			if real_obj_target =='Tableware':
				for obj in obj_state.Tableware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'Drinkware':
				for obj in obj_state.Drinkware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'SmallHandheldObject':
				for obj in obj_state.SmallHandheldObject:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == "Silverware":
				for obj in obj_state.Silverware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target


		if real_parent_target in ["Tables"]:
			for obj in obj_state.Tables:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[parent_target]
						self.cat_equate_dict[obj] = parent_target

		if sliced:
			self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife'] 
			self.cat_equate_dict['ButterKnife'] = 'Knife' #if DeskLamp is found, consider it as FloorLamp
			
		
		actions_dict = {'task_type': task_type, 'list_of_actions': list_of_actions, 'second_object': second_object, 'total_cat2idx': self.total_cat2idx, 'sliced':self.sliced}
		self.print_log('total cat2idx is ', self.total_cat2idx)
		
		self.actions_dict = actions_dict
		#except:
		# self.print_log("Scene pointers exceeded the number of all scenes, for env rank", self.rank)
		# obs = np.zeros(self.obs.shape)
		# info = self.info
		# actions_dict = None

		self.seg.update_agent(self)
		#self.consecutive_interaction_executed = False
		self.last_action_ogn = None

		return obs, info, actions_dict, return_actions

	def load_traj(self, scene_name):
		json_dir = 'src/teach/inference/FILM_refactor_april1/alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
		traj_data = json.load(open(json_dir))
		return traj_data

	def load_next_scene(self, traj_data, image):
		#if load == True:
		#self.scene_pointer += 1
		obs, info, actions_dict, return_actions = self.load_initial_scene(traj_data, image)
		return obs, info, actions_dict, return_actions
		
		#return self.obs, self.info, self.actions_dict


	def setup_scene_step0(self):
		#obs, info = super().setup_scene(traj_data,task_type, r_idx, args, reward_type)
		#print("SUPER SETUP SCENE START")
		#obs, info = super().setup_scene()
		#print("SUPER SETUP SCENE END")

		#Get obs with self.img
		obs = self.get_obs()
		obs, seg_print = self._preprocess_obs(obs)
		print("PREPROCESS OBS END")

		self.obs_shape = obs.shape
		self.obs = obs
		
		# Episode initializations
		map_shape = (self.args.map_size_cm // self.args.map_resolution,
					 self.args.map_size_cm // self.args.map_resolution)
		self.collision_map = np.zeros(map_shape)
		self.visited = np.zeros(map_shape)
		self.col_width = 5
		self.curr_loc = [self.args.map_size_cm/100.0/2.0,
						 self.args.map_size_cm/100.0/2.0, 0.]
		self.seg_print = seg_print

		self.seg.update_agent(self)
		return obs

	# def setup_scene_step_3(self, obs): #Do we need this? Maybe just step is fine
	# 	obs, seg_print = self._preprocess_obs(obs)

	# 	self.obs_shape = obs.shape
	# 	self.obs = obs
		
	# 	# Episode initializations
	# 	map_shape = (args.map_size_cm // args.map_resolution,
	# 				 args.map_size_cm // args.map_resolution)
	# 	self.collision_map = np.zeros(map_shape)
	# 	self.visited = np.zeros(map_shape)
	# 	self.col_width = 5
	# 	self.curr_loc = [args.map_size_cm/100.0/2.0,
	# 					 args.map_size_cm/100.0/2.0, 0.]
	# 	self.last_action_ogn = None
	# 	self.seg_print = seg_print

	# 	return obs, info

	# def get_error(self):
	def _get_approximate_success(self, prev_rgb, frame, action):
		success = False
		if (action in ['MoveAhead_25', "RotateLeft_90", "RotateRight_90", "LookDown_30", "LookUp_30"]): #Max area of just faucet being on is 448 when agent is close
			max_area = self._get_max_area(prev_rgb, frame)
			self.print_log("Action is ", action)
			self.print_log("Max area is ", max_area)
			if max_area>9000:
				success = True
		else:#Interaction actions
			#Get max area inside interaction mask and outside interaction mask
			#For the interaction to succeed, 
			max_area = self._get_max_area(prev_rgb, frame)
			mask_area_interaction_mask = self._get_max_area(prev_rgb, frame, self.interaction_mask)
			below100_mask = np.zeros((300,300)); below100_mask[200:, : ] = 1
			max_area_below100 = self._get_max_area(prev_rgb, frame, below100_mask)
			if action in (["PutObject", "PickupObject"]):
				if max_area_below100 > 100:
					success=True
			elif action == 'SliceObject':
				if mask_area_interaction_mask > 100:
					success=True
			elif action in ['OpenObject', 'CloseObject']:
				if mask_area_interaction_mask > 500:
					success=True
			elif "Toggle" in action:
				success = True
			elif "Pour" in action:
				success = True
			else:
				if not(action in ['LookDown_0', 'LookUp_0']):
					self.print_log("Interaction not in category in _get_approximate_success ", action)
					raise Exception("INTERACTION not in category in _get_approximate_success : " + str(action))
			self.print_log("Approx Success is ", success)
		return success

	#Everything up to va_interact_new
	def semexp_step_internal_right_after_sim_step(self, last_success_special_treatment=None):
		#First call thor_step_internal_right_after_sim_step
		state, info = self.thor_step_internal_right_after_sim_step()
		self.obs_step = state; self.info = info

		#First thing after va_interact_new (the most outer thing in thor)
		#success = CH._get_approximate_success(self.prev_rgb, self.img, action)
		if last_success_special_treatment is None:
			success= self._get_approximate_success(self.prev_rgb, self.img, self.prev_action)
			self.last_success = success
			#if self.PREV_side_step_order == 1:
			#	self.prev_sidestep_success = success

			if not(success) and not(self.prev_action in ['LookUp_0', 'LookDown_0']):
				self.fails_cur =1
			else:
				self.fails_cur =0

		elif last_success_special_treatment == 'NoUpdate':
			#pass
			self.fails_cur =0

		elif last_success_special_treatment == 'FalseUpdate':
			self.last_success = False
			self.fails_cur =0

		if self.PREV_side_step_order == 1:
			self.prev_sidestep_success = self.last_success

		#self.actions = CH._append_to_actseq(success, self.actions, api_action) #This should not be needed
		self.seg.update_agent(self)

		#return obs, rew, done, info, success, a, target_instance, err, api_action #THis success we don't have to care because we will only get it after simulator step now

	#Everything up to va_interact_new
	def semexp_step_internal_right_before_sim_step(self):
		#Final thing before calling step in the simulator
		#There is nothing from va_interact_new
		self.thor_step_internal_right_before_sim_step()


	def _preprocess_obs(self, obs, consecutive_interaction_lower_th=None):
		args = self.args
		obs = obs.transpose(1, 2, 0)
		rgb = obs[:, :, :3]
		depth = obs[:, :, 3:4]	

		sem_seg_pred = self.seg.get_sem_pred(copy.deepcopy(self.img), consecutive_interaction_lower_th) #(300, 300, num_cat)
		#pickle.dump(rgb.astype(np.uint8), open('temp_pickles/sem_seg_rgb.p', 'wb'))
		if self.args.save_pictures:
			if hasattr(self.seg, 'get_sink_mask'):
				SinkMasks = self.seg.get_sink_mask()
				pickle.dump(SinkMasks , open(self.picture_folder_name +"/SinkMasks/" + str(self.steps_taken) +'_SinkMasks.p', 'wb')) 

		if self.args.use_learned_depth: 
			include_mask = np.sum(sem_seg_pred, axis=2).astype(bool).astype(float)
			include_mask = np.expand_dims(np.expand_dims(include_mask, 0), 0)
			include_mask = torch.tensor(include_mask).to(self.depth_gpu)
		
			depth = self.depth_pred_later(include_mask)

		depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

		ds = args.env_frame_width//args.frame_width # Downscaling factor
		if ds != 1:
			rgb = np.asarray(self.res(rgb.astype(np.uint8)))
			depth = depth[ds//2::ds, ds//2::ds]
			sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]

		depth = np.expand_dims(depth, axis=2)
		state = np.concatenate((rgb, depth, sem_seg_pred), axis = 2).transpose(2, 0, 1)

		return state, sem_seg_pred


	def preprocess_obs_success(self, obs, consecutive_interaction_lower_th=None): 
		obs, seg_print =  self._preprocess_obs(obs, consecutive_interaction_lower_th) #= obs, seg_print
		self.obs = obs; self.seg_print = seg_print
		return obs, seg_print


	#Plan Act and preprocess (interact with the mapping module)
	def semexp_plan_act_and_preprocess_before_step(self, planner_inputs, goal_spotted):
		#These are things from plan_act_and_preprocess
		self.pointer = planner_inputs['list_of_actions_pointer']
		traversible, cur_start, cur_start_o = self.get_traversible(planner_inputs)

		self.activate_lookDownUpLeft_count = False
		self.steps += 1
		self.moved_until_visible = False
		self.side_stepped = None
		self.sdroate_direction = None
		self.IN_opp_side_step = False

		self._visualize(planner_inputs)

		self.goal_success = False
		keep_consecutive = False

		self.Return_actions = []

		#If statements and get Return_actions


		self.goal_success = False
		keep_consecutive = False



		if self.side_step_order in [1,2]:
			prev_side_step_order = copy.deepcopy(self.side_step_order)
			self.Return_actions += self.side_step(self.step_dir, cur_start_o, cur_start, traversible)


		elif self.lookDownUpLeft_count in range(1,3):#in 1,2,3
			if self.args.debug_local:
				self.print_log("Tried to lookdownupleft")

			if self.lookDownUpLeft_count ==1:
				action = "LookUp_15"
				cur_hor = np.round(self.camera_horizon, 4)
				self.Return_actions += self.set_back_to_angle(0)
				#self.lookDownUpLeft_count +=1
			elif self.lookDownUpLeft_count ==2:
				#look down back to 45
				cur_hor = np.round(self.camera_horizon, 4)
				self.Return_actions += self.set_back_to_angle(30)
				action = "RotateLeft_90"
				self.Return_actions += self.va_interact_new("RotateLeft_90")
				# self.lookDownUpLeft_count = 0
				

			
		
		elif planner_inputs['consecutive_interaction'] != None:
			if self.args.debug_local:
				self.print_log("consec action")
			
			target_object_type = planner_inputs['consecutive_target']
			
			self.Return_actions += self.consecutive_interaction(planner_inputs['consecutive_interaction'], target_object_type)
			# goal_success = success
			# self.last_action_ogn = planner_inputs['consecutive_interaction']
				
			# self.execute_interaction = False
			
			
			
		elif planner_inputs['consecutive_interaction'] == None and self.execute_interaction:
			#Do the deterministic interaction here
			list_of_actions = planner_inputs['list_of_actions']
			pointer = planner_inputs['list_of_actions_pointer']
			interaction = list_of_actions[pointer][1]
				
			if interaction == None:
				pass
			else:              
				self.Return_actions += self.va_interact_new(interaction, mask=self.interaction_mask)
				# self.execute_interaction = False
				# self.last_action_ogn = interaction
				# if success:
				# 	goal_success = True
				

				
				# obs_temp, seg_print = self.preprocess_obs_success(success, obs)

				# if self.last_action_ogn == "PickupObject" and goal_success:
				# 	self.picked_up = True
				# 	self.picked_up_cat = self.goal_idx
				# 	if self.args.use_sem_seg:
				# 		self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(self.goal_name)
				# 	else:
				# 		self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(self.goal_name)
				# 	obs_temp, seg_print = self.preprocess_obs_success(success, obs)
				# elif self.last_action_ogn == "PutObject" and goal_success:
				# 	self.picked_up = False
				# 	self.picked_up_cat = None
					
				# 	self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame)
				# 	self.picked_up_mask = None
		
				# if self.last_action_ogn == "OpenObject" and success:
				# 	self.open_mask = copy.deepcopy(self.interaction_mask)
				
				# obs = obs_temp

				# self.info = info
	
			
		else:
			if self.prev_number_action !=0:
				number_action = self._plan(planner_inputs, self.steps, goal_spotted, planner_inputs['newly_goal_set'])
				action_dict = {0: "<<stop>>", 1: "MoveAhead_25", 2:"RotateLeft_90", 3:"RotateRight_90", 4: "LookDown_90", 5:"LookDownUpLeft"}
				action = action_dict[number_action]
				if number_action == 0:
					self.prev_number_action = 0
			else:
				number_action = 100 #stop outputted before
				action = "stopAction"
			
			repeated_rotation = (not(self.last_three_sidesteps[0] is None) and not(self.last_three_sidesteps[1] is None)) and \
				"Rotate" in self.last_three_sidesteps[1] and self.prev_sidestep_success == False
			
			
			if self.prev_number_action==0:  #stop outputted now or before
				 if (self.args.approx_error_message and not(self.last_success) and self.last_action_ogn in ["OpenObject", "CloseObject"] ): #or (not(self.args.approx_error_message) and self.last_err=="Object failed to open/close successfully."):
					 self.update_loc(planner_inputs)
					 self.Return_actions += self.move_behind()
					 if self.args.debug_local:
						 self.print_log("Moved behind!")
					 
				 #elif not(self.args.no_rotate_sidestep) and (not(self.last_three_sidesteps[0] is None) and self.prev_sidestep_success == False):
				 elif not(self.args.no_rotate_sidestep) and (not(self.last_three_sidesteps[0] in [None, "RotateRight", "RotateLeft"]) and self.prev_sidestep_success == False):
					 #Rotate to the failed direction 
					 if self.args.debug_local:
						 self.print_log("rotating because sidestepping failed")
					 self.update_loc(planner_inputs)
					 if self.last_three_sidesteps[0] == 'right':
						 self.sdroate_direction = "Right"
					 elif self.last_three_sidesteps[0] == 'left':
						 self.sdroate_direction = "Left"
					 self.Return_actions += self.va_interact_new("Rotate" +self.sdroate_direction+ "_90")
					 self.update_last_three_sidesteps("Rotate" +self.sdroate_direction)
				 
				 elif self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility): #Must be cautious pointers
					 #sidestep
					 self.update_loc(planner_inputs)
					 wd = self.which_direction()
					 if self.args.debug_local:
						 self.print_log("wd is ", wd)
					 if wd <= 100:
						 step_dir = 'left'
						 if self.args.debug_local:
							 self.print_log("sidestepping to left")
					 elif wd > 200:
						 step_dir = 'right'
						 if self.args.debug_local:
							 self.print_log("sidestepping to right")
					 else:
						 step_dir = None
						 if self.args.debug_local:
							 self.print_log("skipping sidestepping")
					 if not(self.args.no_opp_sidestep) and self.last_three_sidesteps == ['left', 'right', 'left'] or self.last_three_sidesteps == ['right', 'left', 'right']:
						 self.opp_side_step = True
						 
						 if step_dir == None:
							 opp_step_dir = None
							 self.Return_actions +=self.va_interact_new("LookUp_0") #pass
						 else:
							 if step_dir == 'left':
								 opp_step_dir = 'right'
							 else:
								 opp_step_dir = 'left'
							 self.Return_actions += self.side_step(opp_step_dir, cur_start_o, cur_start, traversible)
						 self.side_stepped = opp_step_dir
					 else:
						 self.opp_side_step = False
						 if not(step_dir is None):
							 self.Return_actions +=self.side_step(step_dir, cur_start_o, cur_start, traversible)
						 else:
							 self.Return_actions += self.va_interact_new("LookUp_0") #pass
						 self.side_stepped = step_dir
					 if self.args.debug_local:
						 self.print_log("last three side stepped ", self.last_three_sidesteps) 
					 
				 else: #not visible
					 self.moved_until_visible = True
					 if self.args.debug_local:
						 self.print_log("moving until visible")
						 self.print_log("current horizon is ", self.camera_horizon)
						 self.print_log("move until visible order is ", self.move_until_visible_order)
					 self.update_loc(planner_inputs)
					 self.Return_actions += self.move_until_visible()
					 self.mvb_num_in_cycle +=1
					 if self.mvb_num_in_cycle == 12:
						 self.print_log("Went through one cycle of move until visible, step num is ", self.steps_taken)
						 self.mvb_which_cycle +=1
						 self.mvb_num_in_cycle = 0
						 if self.args.delete_from_map_after_move_until_visible:
							 self.prev_number_action = 100 #Release "stop outputted"
			
					
			else:          #stop not ouputted   
				if action == "LookDownUpLeft":
					cur_hor = np.round(self.camera_horizon, 4)
					if abs(cur_hor-60)>5:
						self.Return_actions += self.set_back_to_angle(30)
					else:
						self.Return_actions += self.va_interact_new("LookUp_0")

					#self.lookDownUpLeft_count = 1
					self.activate_lookDownUpLeft_count = True

				else:
					self.Return_actions += self.va_interact_new(action)

		return self.Return_actions

	def semexp_plan_act_and_preprocess_after_step(self, planner_inputs, goal_spotted):
		#The step has been taken now
		#Start with if statements
		if self.PREV_side_step_order in [1,2]:
			self.print_log("Fell into  self.PREV_side_step_order in [1,2]")
			self.print_log("PREV side step order is ", self.PREV_side_step_order)
			#prev_side_step_order = copy.deepcopy(self.side_step_order)
			#obs, rew, done, info, success, target_instance, err = self.side_step(self.step_dir, cur_start_o, cur_start, traversible)
			obs, seg_print = self.preprocess_obs_success(self.obs_step)
			#self.info = info
			self.interaction_mask = None

			if self.PREV_side_step_order == 2:
				self.IN_opp_side_step = self.opp_side_step


			if self.args.use_sem_seg:
				self.print_log("obj type for mask is :", self.goal_idx2cat[self.goal_idx])
				self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(self.goal_idx2cat[self.goal_idx])
			else:
				self.interaction_mask = self.seg.get_instance_mask_from_obj_type(self.goal_idx2cat[self.goal_idx])
			
			if self.PREV_side_step_order == 2:  
				self.execute_interaction =  False

				pointer = planner_inputs['list_of_actions_pointer']
				
				# if self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility):
				# 	visible = True
				# else:
				# 	visible = False
				visible = self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility)

				if not(pointer in self.caution_pointers):
					self.execute_interaction = goal_spotted and visible 
				else: 
					whether_center = self.whether_center()
					self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center 
					if self.IN_opp_side_step:
						self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 
				self.PREV_side_step_order = 0

			else:
				self.execute_interaction =  False

			if self.PREV_side_step_order == 0:
				self.side_stepped = self.step_dir


		elif self.lookDownUpLeft_count in range(1,3):#in 1,2,3
			self.print_log("Fell into self.lookDownUpLeft_count in range(1,3)")
			if self.args.debug_local:
				self.print_log("Tried to lookdownupleft")

			if self.lookDownUpLeft_count ==1:
				action = "LookUp_15"
    
				self.lookDownUpLeft_count +=1
			elif self.lookDownUpLeft_count ==2:

				self.lookDownUpLeft_count = 0
				
			obs, seg_print = self.preprocess_obs_success(self.obs_step)
			#self.last_action_ogn = action
			self.last_action_ogn = self.prev_action
			#self.info = info
			self.execute_interaction = False
			self.interaction_mask = None
			
		
		elif planner_inputs['consecutive_interaction'] != None:
			print("COnsecutive Interaction")
			self.print_log("Fell into planner_inputs['consecutive_interaction'] != None")
			if self.args.debug_local:
				self.print_log("consec action")
			
			#target_object_type = planner_inputs['consecutive_target']
			
			#obs, rew, done, info, success, err= self.consecutive_interaction(planner_inputs['consecutive_interaction'], target_object_type)
			self.goal_success = self.last_success
			self.last_action_ogn = planner_inputs['consecutive_interaction']
			self.execute_interaction = False

			obs_temp, seg_print = self.preprocess_obs_success(self.obs_step)#,consecutive_interaction_lower_th=0.6)

			if self.last_action_ogn == "PickupObject" and self.goal_success:
				self.picked_up = True
				self.picked_up_cat = self.goal_idx
				if self.args.use_sem_seg:
					self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(self.goal_name)
				else:
					self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(self.goal_name)
				obs_temp, seg_print = self.preprocess_obs_success(self.obs_step)
			elif self.last_action_ogn == "PutObject" and self.goal_success:
				self.picked_up = False
				self.picked_up_cat = None
				
				self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.img)
				self.picked_up_mask = None
	
			if self.last_action_ogn == "OpenObject" and self.last_success:
				self.open_mask = copy.deepcopy(self.interaction_mask)
			
			obs = obs_temp
			if self.last_action_ogn == "SliceObject" and self.goal_success:
				self.first_goal_sliced_state = True
		
			
			
			
		elif planner_inputs['consecutive_interaction'] == None and self.execute_interaction:
			self.print_log("Fell into planner_inputs['consecutive_interaction'] == None and self.execute_interaction")
			#Do the deterministic interaction here
			#list_of_actions = planner_inputs['list_of_actions']
			#pointer = planner_inputs['list_of_actions_pointer']
			#interaction = list_of_actions[pointer][1]
			list_of_actions = planner_inputs['list_of_actions']
			pointer = planner_inputs['list_of_actions_pointer']
			interaction = list_of_actions[pointer][1]
			print("Interaction is ", interaction)
				
			if interaction == None:
				pass
			else:              
				#obs, rew, done, info, success, _, target_instance, err, _ = self.va_interact_new(interaction, self.interaction_mask)
				self.execute_interaction = False
				self.last_action_ogn = interaction
				self.goal_success = self.last_success
				self.print_log("Goal success was ", self.goal_success)
				
				obs_temp, seg_print = self.preprocess_obs_success(self.obs_step)

				if self.last_action_ogn == "PickupObject" and self.goal_success:
					self.picked_up = True
					self.picked_up_cat = self.goal_idx
					if self.args.use_sem_seg:
						self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(self.goal_name)
					else:
						self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(self.goal_name)
					obs_temp, seg_print = self.preprocess_obs_success(self.obs_step)
				elif self.last_action_ogn == "PutObject" and self.goal_success:
					self.picked_up = False
					self.picked_up_cat = None
					
					self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.img)
					self.picked_up_mask = None
		
				if self.last_action_ogn == "OpenObject" and self.last_success:
					self.open_mask = copy.deepcopy(self.interaction_mask)
				
				obs = obs_temp

				#self.info = info
			if self.last_action_ogn == "SliceObject" and self.goal_success:
				self.first_goal_sliced_state = True
			
		else:
			if self.activate_lookDownUpLeft_count:
				self.lookDownUpLeft_count = 1
			#This block was already done before
			####################
			self.print_log("Fell into else")
			obs, seg_print = self.preprocess_obs_success(self.obs_step)
			if self.args.use_sem_seg:
				self.print_log("obj type for mask is :", self.goal_idx2cat[self.goal_idx])
				if not(self.first_goal_sliced_state):
					self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(self.goal_idx2cat[self.goal_idx])
				else:
					self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_smallest_only(self.goal_idx2cat[self.goal_idx])
			else:
				self.interaction_mask = self.seg.get_instance_mask_from_obj_type(self.goal_idx2cat[self.goal_idx])
			
			visible = self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility)
			
			#self.last_action_ogn = action
			self.last_action_ogn = self.prev_action
			#self.info = info
			
			list_of_actions = planner_inputs['list_of_actions']
			pointer = planner_inputs['list_of_actions_pointer']
			interaction = list_of_actions[pointer][1]

			if self.args.stricter_visibility <= 1.5:
				if self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility) and self.whether_center():
					self.prev_number_action == 0

			pointer = planner_inputs['list_of_actions_pointer']
			
			self.execute_interaction = False
			if not(pointer in self.caution_pointers):
				self.execute_interaction = goal_spotted and visible 
			else: #caution pointers
				whether_center = self.whether_center()
				self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center 
			
				if self.IN_opp_side_step:
					self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 

		delete_lamp = (self.mvb_num_in_cycle !=0)  and (self.goal_name == 'FloorLamp') and (self.action_received == "ToggleObjectOn")
		if self.args.no_delete_lamp:
			delete_lamp = False

		if not(self.moved_until_visible):
			self.mvb_num_in_cycle = 0
			self.mvb_which_cycle = 0



		self.rotate_aftersidestep =self.sdroate_direction
		next_step_dict = {'keep_consecutive': False, 'view_angle': self.camera_horizon, \
								  'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken, 'broken_grid':self.broken_grid, 'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['instance_id']): self.actions[:1000]},
								  'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'move_until_visible_cycled': self.mvb_which_cycle != 0 and self.mvb_num_in_cycle==0, 'delete_lamp': delete_lamp,
								  'fails_cur': self.fails_cur}
		#self.last_err = err
		if self.side_step_order ==0 and self.side_stepped is None:
			self.prev_sidestep_success = True
			self.update_last_three_sidesteps(self.side_stepped)
				
		# if self.args.debug_local:
		# 	if err!="":
		# 		self.print_log("step: " , str(self.steps_taken), ", err is ", err)
		# 		self.print_log("action taken in step ", str(self.steps_taken), ": ",)

		#self.info = info

		
		list_of_actions = planner_inputs['list_of_actions']
		pointer = planner_inputs['list_of_actions_pointer']
		self.print_log("pointer ", pointer)
		self.print_log("len list of actions ", len(list_of_actions))
		self.print_log("Goal success ", self.goal_success)
		if self.goal_success and pointer +1 < len(list_of_actions):
			self.print_log("pointer increased goal name ", list_of_actions[pointer+1])
		
		self.print_log("Self.execute_interaction is ", self.execute_interaction)
		#return obs, info, goal_success, next_step_dict
		return self.obs, self.info, self.goal_success, next_step_dict


	###############################################################################################
	#All the actions
	def va_interact_new(self, action, last_success_special_treatment = None, mask=None):		
		 print("Returning ", [(action, mask, last_success_special_treatment)])  
		 return [(action, mask, last_success_special_treatment)]

	def set_back_to_angle(self, angle_arg, last_success_special_treatment = None):
		Return_actions = []
		look_failure = False
		view_angle_copy = copy.deepcopy(int(self.camera_horizon) )
		if abs(view_angle_copy - angle_arg)> 5:
			if view_angle_copy> angle_arg:
				#Looking down like 60 degrees
				#Look up until 45
				times_30  = int((view_angle_copy-angle_arg)/30)
				for i in range(times_30):
					Return_actions += self.va_interact_new("LookUp_30", last_success_special_treatment =last_success_special_treatment)
					# if self.args.look_no_repeat and not(success): #-> Took out for TEACH
					# 	look_failure = True
					# 	break
				#if not(look_failure) and abs(int(self.camera_horizon)  - angle_arg) > 5:
				if abs(int(self.camera_horizon)  - angle_arg) > 5:
					angle = (view_angle_copy- angle_arg) - 30 * times_30
					Return_actions +=self.va_interact_new("LookUp_" + str(angle), last_success_special_treatment =last_success_special_treatment)
					

			else:
				times_30 = int((angle_arg-view_angle_copy)/30)
				for i in range(times_30):
					Return_actions +=self.va_interact_new("LookDown_30", last_success_special_treatment =last_success_special_treatment)
					#print("Looked down once") #-> Took out for TEACH
					# if self.args.look_no_repeat and not(success):
					# 	look_failure = True
					# 	break
				#if not(look_failure) and abs(int(self.camera_horizon)  - angle_arg) > 5:
				if abs(int(self.camera_horizon)  - angle_arg) > 5:
					angle = (angle_arg - view_angle_copy) - 30 * times_30
					Return_actions += self.va_interact_new("LookDown_" + str(angle), last_success_special_treatment =last_success_special_treatment)

			return Return_actions

		else:
			return self.va_interact_new("LookUp_0", last_success_special_treatment =last_success_special_treatment)

	def move_until_visible(self):
		Return_actions = []
		if not(self.rotate_aftersidestep is None) and abs(int(self.camera_horizon)  - 0) <5 : #rotated after sidestep but no longer visible
			return self.set_back_to_angle(30)
		else:
			self.move_until_visible_order = self.move_until_visible_order %8
			order = self.move_until_visible_order
			if  abs(int(self.camera_horizon)  - 60) <5: 
				if order <=3: #0,1,2,3
					action = "RotateLeft_90"
					Return_actions += self.va_interact_new(action)
				elif order >3: #4,5,6,7
					#self.consecutive_steps = True
					action = "RotateLeft_90"
					Return_actions += self.va_interact_new(action)
					Return_actions += self.set_back_to_angle(0, last_success_special_treatment = "NoUpdate")
					#self.final_sidestep = True
					if Return_actions == []:
						Return_actions += self.va_interact_new("LookUp_0", last_success_special_treatment = "NoUpdate")
					self.print_log("RETURN ACTIONS OF move_until_visible are ", Return_actions)
					self.print_log("Move until visible horizon is ", self.camera_horizon)
					#self.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon'])

					#self.consecutive_steps = False
					#self.final_sidestep = False
				self.move_until_visible_order +=1
				
			else:
				self.print_log("Move until visible horizon is ", self.camera_horizon)
				#self.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon'])
				Return_actions += self.set_back_to_angle(30)
		#try:
		#	success = success1
		#except:
		#	self.print_log("move until visible broken!")
		return Return_actions

	def side_step(self, step_dir, cur_start_o, cur_start, traversible):
		Return_actions = []
		self.print_log("side step called, order: ", self.side_step_order)
		self.PREV_side_step_order = self.side_step_order
		if self.side_step_order ==0:
			self.step_dir = step_dir
			if step_dir == 'left':
				#turn left, moveforward, turn right
				Return_actions += self.va_interact_new("RotateLeft_90")

			elif step_dir == 'right':
				Return_actions += self.va_interact_new("RotateRight_90")
			else:
				raise Exception("Invalid sidestep direction")
			self.side_step_order =1

			self.update_last_three_sidesteps(step_dir)	
		
		elif self.side_step_order ==2:
			if step_dir == 'right':
				#turn left, moveforward, turn right
				Return_actions += self.va_interact_new("RotateLeft_90")

			elif step_dir == 'left':
				Return_actions += self.va_interact_new("RotateRight_90")
			else:
				print("exception raised")
				raise Exception("Invalid sidestep direction")
			self.side_step_order =0
		#Move ahead
		elif self.side_step_order ==1:
			if hasattr(self.args, 'check_before_sidestep') and self.args.check_before_sidestep:
				self.print_log("checking consec moving!")
				xy = CH._which_direction(cur_start_o)
				whether_move = CH._check_five_pixels_ahead_map_pred_for_moving(self.args, traversible, cur_start,  xy)
			else:
				whether_move = True
			if not(whether_move):
				#self.print_log("not move because no space!")
				Return_actions +=self.va_interact_new("LookDown_0", last_success_special_treatment = "FalseUpdate")
				#success = False
				#if not(success):
				#	self.print_log("side step  move prevented from check!")
			else:
				Return_actions +=self.va_interact_new("MoveAhead_25")
				#if not(success):
				#	self.print_log("side step moved and failed!")
			self.side_step_order =2
			#self.prev_sidestep_success = success
		else:
			print("exception raised")
			raise Exception("Invalid sidestep order")

		return Return_actions

	def move_behind(self, cur_start_o=None, cur_start=None, traversible=None):
		Return_actions = []
		#self.consecutive_steps = True
		#self.final_sidestep = False
		Return_actions += self.va_interact_new("RotateLeft_90")
		Return_actions += self.va_interact_new("RotateLeft_90")
		
		Return_actions += self.va_interact_new("MoveAhead_25")
		Return_actions += self.va_interact_new("RotateLeft_90")
	
		Return_actions += self.va_interact_new("RotateLeft_90")
		return Return_actions

	# def right_after_consecutive_interaction(self, interaction):
	# 	self.last_action_ogn = interaction

	# 	if self.last_action_ogn == "PickupObject" and self.last_success:
	# 		self.picked_up = True
	# 		self.picked_up_cat = self.goal_idx
	# 		if self.args.use_sem_seg:
	# 			self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(self.goal_name)
	# 		else:
	# 			self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(self.goal_name)

	# 		#obs_temp, seg_print = self.preprocess_obs_success(success, obs)

	# 	elif self.last_action_ogn == "PutObject" and self.last_success:
	# 		self.picked_up = False
	# 		self.picked_up_cat = None
			
	# 		self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.img)
	# 		self.picked_up_mask = None
			
	# 	#obs = obs_temp #Should this be self.obs?

	# 	if self.last_action_ogn == "OpenObject" and self.last_success:
	# 		self.open_mask = copy.deepcopy(self.interaction_mask)
		
		
		#self.info = info

	def consecutive_interaction(self, interaction, target_object_type):
		Return_actions = []
		if interaction == "PutObject" and self.last_action_ogn == "OpenObject":
			self.interaction_mask =self.open_mask
		elif interaction == "CloseObject":
			self.interaction_mask =self.open_mask
		elif self.args.use_sem_seg:
			self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(target_object_type)
		else:
			self.interaction_mask = self.seg.get_instance_mask_from_obj_type(target_object_type)

		Return_actions += self.va_interact_new(interaction, mask =self.interaction_mask)
		#self.consecutive_interaction_executed = True 

		# if self.last_action_ogn == "PickupObject" and not(success):
		# 	self.interaction_mask = self.put_rgb_mask
		# 	obs, rew, done, info, success, _, target_instance, err, _ = self.va_interact_new(interaction, self.interaction_mask) 
		# 	self.last_action_ogn = interaction

		# obs_temp, seg_print = self.preprocess_obs_success(success, obs)
		# if self.last_action_ogn == "PickupObject" and success:
		# 	self.picked_up = True
		# 	self.picked_up_cat = self.goal_idx
		# 	if self.args.use_sem_seg:
		# 		self.picked_up_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(self.goal_name)
		# 	else:
		# 		self.picked_up_mask = self.seg.get_instance_mask_from_obj_type_largest(self.goal_name)

		# 	obs_temp, seg_print = self.preprocess_obs_success(success, obs)

		######################################################
		##### Moved these to right after consecutive interaction
		# if self.last_action_ogn == "PutObject" and success:
		# 	self.picked_up = False
		# 	self.picked_up_cat = None
			
		# 	self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.event.frame)
		# 	self.picked_up_mask = None
			
		# obs = obs_temp

		# if self.last_action_ogn == "OpenObject" and success:
		# 	self.open_mask = copy.deepcopy(self.interaction_mask)
		
		
		# self.info = info

		#preprocess obs
		return Return_actions


	################
	####Control
	################
	def update_last_three_sidesteps(self, new_sidestep):
		self.last_three_sidesteps = self.last_three_sidesteps[:2]
		self.last_three_sidesteps = [new_sidestep] + self.last_three_sidesteps

	def update_loc(self, planner_inputs):
		self.last_loc = self.curr_loc

		# Get pose prediction and global policy planning window
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
				planner_inputs['pose_pred']
		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2) 

		# Get curr loc
		self.curr_loc = [start_x, start_y, start_o]
		
		r, c = start_y, start_x
		start = [int(r * 100.0/self.args.map_resolution - gx1),
				 int(c * 100.0/self.args.map_resolution - gy1)]
		map_pred = np.rint(planner_inputs['map_pred'])
		start = pu.threshold_poses(start, map_pred.shape)

		self.visited[gx1:gx2, gy1:gy2][start[0]-0:start[0]+1,
									   start[1]-0:start[1]+1] = 1
		
	def which_direction(self):
		if self.interaction_mask is None:
			return 150
		widths = np.where(self.interaction_mask !=0)[1] 
		center = np.mean(widths)
		return center

	def whether_center(self):
		if self.interaction_mask is None:
			return False
		wd = self.which_direction()
		if np.sum(self.interaction_mask) == 0:#if target_instance not in frame
			return False
		elif wd >= 65 and wd<=235:
			return True
		else:
			return False

	def get_traversible(self, planner_inputs):
		args = self.args
		map_pred = np.rint(planner_inputs['map_pred'])
		grid = map_pred
		

		# Get pose prediction and global policy planning window
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
				planner_inputs['pose_pred']
		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
		planning_window = [gx1, gx2, gy1, gy2]

		# Get curr loc
		r, c = start_y, start_x
		start = [int(r * 100.0/self.args.map_resolution - gx1),
				 int(c * 100.0/self.args.map_resolution - gy1)]
		start = pu.threshold_poses(start, map_pred.shape)

		#Get traversible
		def add_boundary(mat, value=1):
			h, w = mat.shape
			new_mat = np.zeros((h+2,w+2)) + value
			new_mat[1:h+1,1:w+1] = mat
			return new_mat
		
		def delete_boundary(mat):
			new_mat = copy.deepcopy(mat)
			return new_mat[1:-1,1:-1]
		
		
		[gx1, gx2, gy1, gy2] = planning_window

		x1, y1, = 0, 0
		x2, y2 = grid.shape

		if self.args.obstacle_selem >= 1:
			traversible = skimage.morphology.binary_dilation(
						grid[x1:x2, y1:y2],
						self.selem) != True
		else:
			traversible = grid[x1:x2, y1:y2]
		traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
					int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

		if not(traversible[start[0], start[1]]):
			print("Not traversible, step is  ", self.steps_taken)

		traversible = add_boundary(traversible)

		# obstacle dilation
		traversible = 1 - traversible
		selem = skimage.morphology.disk(1)
		traversible = skimage.morphology.binary_dilation(
						traversible, selem) != True
		
		
		traversible = traversible * 1.

		return traversible, start, start_o

	def _plan(self, planner_inputs, step, goal_spotted, newly_goal_set):
		"""Function responsible for planning

		Args:
			planner_inputs (dict):
				dict with following keys:
					'map_pred'  (ndarray): (M, M) map prediction
					'goal'      (ndarray): (M, M) goal locations
					'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
								 and planning window (gx1, gx2, gy1, gy2)
					'found_goal' (bool): whether the goal object is found

		Returns:
			action (int): action id
		"""
		if newly_goal_set:
			self.action_5_count = 0
		
		args = self.args


		self.last_loc = self.curr_loc

		# Get Map prediction
		map_pred = np.rint(planner_inputs['map_pred'])
		goal = planner_inputs['goal']

		# Get pose prediction and global policy planning window
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
				planner_inputs['pose_pred']
		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
		planning_window = [gx1, gx2, gy1, gy2]

		# Get curr loc
		self.curr_loc = [start_x, start_y, start_o]
		r, c = start_y, start_x
		start = [int(r * 100.0/args.map_resolution - gx1),
				 int(c * 100.0/args.map_resolution - gy1)]
		start = pu.threshold_poses(start, map_pred.shape)

		self.visited[gx1:gx2, gy1:gy2][start[0]-0:start[0]+1,
									   start[1]-0:start[1]+1] = 1
		
		xy = CH._which_direction(start_o)
		xy = np.array(xy)

		if self.last_action_ogn == "MoveAhead_25":
			x1, y1, t1 = self.last_loc
			x2, y2, t2 = self.curr_loc
			buf = 4
			length = 2

			if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
				self.col_width += 5
				self.col_width = min(self.col_width, 15)
			else:
				self.col_width = 5

			dist = pu.get_l2_distance(x1, x2, y1, y2)
			if self.last_action_ogn == "MoveAhead_25":
				col_threshold = args.collision_threshold
			elif "LookUp" in self.last_action_ogn:
				col_threshold = args.collision_threshold

			if dist < col_threshold: #Collision
				width = self.col_width
				for i in range(length):
					for j in range(width):
						wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
										(j-width//2) * np.sin(np.deg2rad(t1)))
						wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
										(j-width//2) * np.cos(np.deg2rad(t1)))
						r, c = wy, wx
						r, c = int(round(r*100/args.map_resolution)), \
							   int(round(c*100/args.map_resolution))
						[r, c] = pu.threshold_poses([r, c],
									self.collision_map.shape)
						self.collision_map[r,c] = 1

		self.goal_visualize = goal
		stg, stop, whether_real_goal = self._get_stg(map_pred, start, np.copy(goal),
									planning_window, planner_inputs['found_goal'], xy.tolist(), step, planner_inputs['exp_pred'], goal_spotted, newly_goal_set, planner_inputs['list_of_actions_pointer'])

		# Deterministic Local Policy
		if stop and whether_real_goal:
			action = 0

		elif stop:
			if self.action_5_count < 4:
				action = 5 #lookdown, lookup, left
				self.action_5_count +=1
			else:
				action = 2
		else:
			(stg_x, stg_y) = stg
			angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
												stg_y - start[1]))
			angle_agent = (start_o)%360.0
			if angle_agent > 180:
				angle_agent -= 360

			relative_angle = (angle_agent - angle_st_goal)%360.0
			if relative_angle > 180:
				relative_angle -= 360

			if relative_angle > 45:
				action = 3 # Right
			elif relative_angle < -45:
				action = 2 # Left
			else:
				action = 1

		return action     

	def _get_stg(self, grid, start, goal, planning_window, found_goal, xy_forward, step, explored, goal_found, newly_goal_set, pointer):
		def add_boundary(mat, value=1):
			h, w = mat.shape
			new_mat = np.zeros((h+2,w+2)) + value
			new_mat[1:h+1,1:w+1] = mat
			return new_mat
		
		def delete_boundary(mat):
			new_mat = copy.deepcopy(mat)
			return new_mat[1:-1,1:-1]
		
		if goal.shape == (240, 240):
			goal = add_boundary(goal, value=0)
		original_goal = copy.deepcopy(goal)
		
		[gx1, gx2, gy1, gy2] = planning_window

		x1, y1, = 0, 0
		x2, y2 = grid.shape

		if self.args.obstacle_selem >= 1:
			traversible = skimage.morphology.binary_dilation(
						grid[x1:x2, y1:y2],
						self.selem) != True
		else:
			traversible = grid[x1:x2, y1:y2]
		traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
					int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

		traversible = add_boundary(traversible)

		# obstacle dilation
		traversible = 1 - traversible
		selem = skimage.morphology.disk(1)#change to 5?
		#selem = skimage.morphology.disk(1)
		traversible = skimage.morphology.binary_dilation(
						traversible, selem) != True
		
		
		traversible = traversible * 1.
	   
		goal_shape = goal.shape
		if newly_goal_set:
			if self.args.debug_local:
				self.print_log("newly goal set")
			self.prev_wall_goal = None
			self.dilation_deg = 0
			
		centers = []
		if len(np.where(goal !=0)[0]) > 1:
			if self.args.debug_local:
				self.print_log("Center done")
			goal, centers = CH._get_center_goal(goal, pointer)
			
		goal_copy = copy.deepcopy(goal)
		goal_to_save = copy.deepcopy(goal)
		 
		
		
		planner = FMMPlanner(self, traversible, self.args, step_size=self.args.step_size)
		
		if not(self.prev_wall_goal is None) and not(goal_found):
			if self.args.debug_local:
				self.print_log("wall goal")
			goal = self.prev_wall_goal
			self.goal_visualize = delete_boundary(goal)
			
		if self.dilation_deg!=0: 
			if self.args.debug_local:
				self.print_log("dilation added")
			goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
			
		if self.prev_wall_goal is None and self.dilation_deg==0:
			if self.args.debug_local:
				self.print_log("None of that")
			else:
				pass
			
		if goal_found:
			if self.args.debug_local:
				self.print_log("goal found!")
			try:
				goal = CH._block_goal(centers, goal, original_goal, goal_found)
			except:
				np.random.seed(self.steps_taken)
				w_goal = np.random.choice(240)
				np.random.seed(self.steps_taken + 1000)
				h_goal = np.random.choice(240)
				goal[w_goal, h_goal] = 1

		try:
			planner.set_multi_goal(goal)
		except:
			#Just set a random place as goal
			np.random.seed(self.steps_taken)
			w_goal = np.random.choice(240)
			np.random.seed(self.steps_taken + 1000)
			h_goal = np.random.choice(240)
			goal[w_goal, h_goal] = 1
			planner.set_multi_goal(goal)
		
		planner_broken, where_connected = CH._planner_broken(planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited)


		if self.args.debug_local:
			self.print_log("planner broken is ", planner_broken)
		
		d_threshold = 60
		
		cur_wall_goal = False
		if planner_broken and self.steps>1:
			if self.args.debug_local:
				self.print_log("Planner broken!")
			
			#goal in obstruction case 
			if goal_found or self.args.use_sem_policy:
				if self.args.debug_local:
					if not(goal_found):
						self.print_log("Goal in obstruction")
					else:
						self.print_log("Goal found, goal in obstruction")      
					self.print_log("Really broken?", CH._planner_broken(planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited)[0])
				while CH._planner_broken(planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited)[0] and (self.dilation_deg<d_threshold):
					goal = copy.deepcopy(goal_copy)
					self.dilation_deg += 1
					goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
					if goal_found: #add original goal area
						goal = CH._block_goal(centers,  goal, original_goal, goal_found)
						if np.sum(goal) ==0:
							goal = goal_copy
					try:
						planner.set_multi_goal(goal)
					except:
						#Just set a random place as goal
						np.random.seed(self.steps_taken)
						w_goal = np.random.choice(240)
						np.random.seed(self.steps_taken + 1000)
						h_goal = np.random.choice(240)
						goal[w_goal, h_goal] = 1
						planner.set_multi_goal(goal)
					if self.args.debug_local:
						self.print_log("dilation is ", self.dilation_deg)
						self.print_log("Sanity check passed in loop is ", CH._planner_broken(planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited)[0])

				if self.dilation_deg == d_threshold:
					if self.args.debug_local:
						self.print_log("Switched to goal in wall after dilation>45")
					self.dilation_deg = 0

					np.random.seed(self.steps_taken)
					random_goal_idx =  np.random.choice(len(where_connected[0]))#choose a random goal among explored area 
					random_goal_ij = (where_connected[0][random_goal_idx], where_connected[1][random_goal_idx])
					goal = np.zeros(goal_shape)
					goal[random_goal_ij[0], random_goal_ij[1]] = 1
					self.prev_wall_goal = goal
					self.goal_visualize = delete_boundary(goal)
					try:
						planner.set_multi_goal(goal)
					except:
						#Just set a random place as goal
						np.random.seed(self.steps_taken)
						w_goal = np.random.choice(240)
						np.random.seed(self.steps_taken + 1000)
						h_goal = np.random.choice(240)
						goal[w_goal, h_goal] = 1
						planner.set_multi_goal(goal)
					cur_wall_goal = True
				
			else:
				if self.args.debug_local:
					self.print_log("Goal in wall, or goal in obstruction althuogh not goal found")
				np.random.seed(self.steps_taken)
				random_goal_idx =  np.random.choice(len(where_connected[0]))#choose a random goal among explored area 
				random_goal_ij = (where_connected[0][random_goal_idx], where_connected[1][random_goal_idx])
				goal = np.zeros(goal_shape)
				goal[random_goal_ij[0], random_goal_ij[1]] = 1
				self.prev_wall_goal = goal
				self.goal_visualize = delete_boundary(goal)
				try:
					planner.set_multi_goal(goal)
				except:
					#Just set a random place as goal
					np.random.seed(self.steps_taken)
					w_goal = np.random.choice(240)
					np.random.seed(self.steps_taken + 1000)
					h_goal = np.random.choice(240)
					goal[w_goal, h_goal] = 1
					planner.set_multi_goal(goal)
				cur_wall_goal = True
			
				
			if self.args.debug_local:
				self.print_log("Sanity check passed is ", CH._planner_broken(planner.fmm_dist, goal, traversible, start, self.steps_taken, self.visited)[0])
			
		if self.args.save_pictures:
			cv2.imwrite(self.picture_folder_name +"fmm_dist/"+ "fmm_dist_" + str(self.steps_taken) + ".png", planner.fmm_dist) 

			# if not(self.depth is None):
			# 	gt_depth = self.depth/ self.depth.max()
			# 	p = np.zeros((300,300, 3)); p[:, :, 0] = gt_depth; p[:, :, 1] = gt_depth; p[:, :, 2] = gt_depth
			# 	gt_depth = p * 255
			# 	gt_depth = gt_depth.astype('uint8')
			# 	cv2.imwrite(self.picture_folder_name +"gt_depth/"+ "gt_depth_" + str(self.steps_taken) + ".png", gt_depth)
		

		state = [start[0] - x1 + 1, start[1] - y1 + 1]
		decrease_stop_cond =0
		if self.dilation_deg >= 6:
			decrease_stop_cond = self.args.decrease_stop_cond #0.2 #decrease to 0.2 (7 grids until closest goal)
		stg_x, stg_y, _, stop = planner.get_short_term_goal(state, found_goal = found_goal, decrease_stop_cond=decrease_stop_cond)
		self.fmm_dist = planner.fmm_dist
		#self.print_log("fmm dist is ", self.fmm_dist)

		stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
		
		self.closest_goal = CH._get_closest_goal(start, goal)

		
		whether_real_goal = found_goal and not(cur_wall_goal)

		self.prev_goal_cos = copy.deepcopy(goal_to_save)
		self.prev_step_goal_cos = copy.deepcopy(goal)
		
		return (stg_x, stg_y), stop, whether_real_goal


	################
	####Resetting goal and everything
	################
	def reset_goal(self, truefalse, goal_name, consecutive_interaction):
		self.Return_actions = []
		#if truefalse == True:
		self.goal_name = goal_name
		if "Sliced" in goal_name :
			self.cur_goal_sliced = self.total_cat2idx[goal_name.replace('Sliced', '')]
		else:
			self.cur_goal_sliced = None
		self.goal_idx = self.total_cat2idx[goal_name]
		self.info['goal_cat_id'] = self.goal_idx
		self.info['goal_name'] = self.goal_name
		
		self.prev_number_action = None
		self.dilation_deg = 0
		self.prev_wall_goal = None
		self.repeating_sidestep = 0
		self.where_block = []

		self.mvb_num_in_cycle = 0
		self.mvb_which_cycle = 0


		self.cur_goal_sem_seg_threshold_small = self.args.sem_seg_threshold_small
		self.cur_goal_sem_seg_threshold_large = self.args.sem_seg_threshold_large
		
		if abs(int(self.camera_horizon)  - 60) >5 and consecutive_interaction is None:
			self.Return_actions = self.set_back_to_angle(30)
		
		self.info['view_angle'] = self.camera_horizon
		
		return self.info, self.Return_actions
		
	def reset_total_cat_new(self, categories_in_inst):
		total_cat2idx = {}

		total_cat2idx["Knife"] =  len(total_cat2idx)
		total_cat2idx["SinkBasin"] =  len(total_cat2idx)
		if self.args.use_sem_policy:
			for obj in constants.map_save_large_objects:
				if not(obj == "SinkBasin"):
					total_cat2idx[obj] = len(total_cat2idx)


		start_idx = len(total_cat2idx)  # 1 for "fake"
		start_idx += 4 *self.rank
		cat_counter = 0
		#assert len(categories_in_inst) <=6
		#Keep total_cat2idx just for 
		for v in categories_in_inst:
			if not(v in total_cat2idx):
				total_cat2idx[v] = start_idx+ cat_counter
				cat_counter +=1 
		
		total_cat2idx["None"] = 11
		if self.args.use_sem_policy:
			total_cat2idx["None"] = total_cat2idx["None"] + 23
		self.total_cat2idx = total_cat2idx
		self.goal_idx2cat = {v:k for k, v in self.total_cat2idx.items()}
		print("self.goal_idx2cat is ", self.goal_idx2cat)
		self.cat_list = categories_in_inst
		self.args.num_sem_categories = 12 #1 + 1 + 1 + 5 * self.args.num_processes 
		if self.args.use_sem_policy:
			self.args.num_sem_categories = self.args.num_sem_categories + 23

	##################
	## Vision
	################
	def is_visible_from_mask(self, mask, stricer_visibility_dist=1.5):
		if not(self.args.learned_visibility):
			if mask is None or np.sum(mask) == 0:
				return False
			#min_depth = np.min(self.event.depth_frame[np.where(mask)])
			min_depth = np.min(self.depth[np.where(mask)])
			print("min depth is ", min_depth)
			return min_depth <= stricer_visibility_dist

		else:
			if mask is None or np.sum(mask) == 0:
				return False
			min_depth = np.min(self.learned_depth_frame[np.where(mask)])
			print("min depth predicted is ", min_depth)
			#min_depth = np.min(self.depth[np.where(mask)])
			#print("actual min depth actual is ", min_depth)
			return min_depth <= stricer_visibility_dist

	def depth_pred_later(self, sem_seg_pred):
		rgb = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)#shape (h, w, 3)
		#pickle.dump(rgb, open('temp_pickles/depth_rgb.p', 'wb'))
		rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255
		
		if abs(self.camera_horizon - 0) <5:
			_, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float())

		else:
			_, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())

		if abs(self.camera_horizon - 0) <5:
			include_mask_prop=self.args.valts_trustworthy_obj_prop0
		else:
			include_mask_prop=self.args.valts_trustworthy_obj_prop
		depth_img = pred_depth.get_trustworthy_depth(max_conf_int_width_prop=self.args.valts_confidence_prop, include_mask=sem_seg_pred, confidence=self.args.valts_confidence, include_mask_prop=include_mask_prop) #default is 1.0
		depth_img = depth_img.squeeze().detach().cpu().numpy()
		if hasattr(self.args, 'ignore_below'):
			#wheres_new = (wheres[0]+280, wheres[1])
			wheres = np.where(depth_img[self.args.ignore_below:, : ] >= self.args.ignore_below_thr)#wheres = np.where(depth_img[self.args.ignore_below:, : ] >= 1.0)
			wheres_new = (wheres[0]+self.args.ignore_below, wheres[1])
			depth_img[wheres_new ] = 0.0
			#wheres = np.where(depth_img[100:, : ] >= 1.65)
			#wheres_new = (wheres[0]+100, wheres[1])
			#depth_img[wheres_new ] = 0.0
		self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
		self.learned_depth_frame = self.learned_depth_frame.reshape((50,300,300))
		self.learned_depth_frame = 5 * 1/50 * np.argmax(self.learned_depth_frame, axis=0) #Now shape is (300,300) 
		del pred_depth
		depth = depth_img

		depth = np.expand_dims(depth, 2)
		return depth

	

	def _preprocess_depth(self, depth, min_d, max_d):
		depth = depth[:, :, 0]*1 #shape (h,w)

		if self.picked_up:
			mask_err_below = depth <0.5
			if not(self.picked_up_mask is None):
				mask_picked_up = self.picked_up_mask == 1
				depth[mask_picked_up] = 100.0
		else:
			mask_err_below = depth <0.0
		depth[mask_err_below] = 100.0
		
		depth = depth * 100
		return depth

	##################
	## Visualize and evaluate
	################
	def _visualize(self, inputs):
		args = self.args

		map_pred = inputs['map_pred']
		exp_pred = inputs['exp_pred']
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
				inputs['pose_pred']

		r, c = start_y, start_x
		start = [int(r * 100.0/args.map_resolution - gx1),
				 int(c * 100.0/args.map_resolution - gy1)]
		start = pu.threshold_poses(start, map_pred.shape)

		#goal = inputs['goal']
		if self.steps <=1:
			goal = inputs['goal']
		else:
			goal = self.goal_visualize
		sem_map = inputs['sem_map_pred']

		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)

		grid = np.rint(map_pred)
		explored = np.rint(exp_pred)

		sem_map += 5

		if self.args.ground_truth_segmentation:
			no_cat_mask = sem_map == 5 + args.num_sem_categories -1
		else:
			no_cat_mask = sem_map == 20
		map_mask = np.rint(map_pred) == 1
		map_mask = np.logical_or(map_mask, self.collision_map==1)
		exp_mask = np.rint(exp_pred) == 1
		vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

		sem_map[no_cat_mask] = 0
		m1 = np.logical_and(no_cat_mask, exp_mask)
		sem_map[m1] = 2

		m2 = np.logical_and(no_cat_mask, map_mask)
		sem_map[m2] = 1

		sem_map[vis_mask] = 3

		curr_mask = np.zeros(vis_mask.shape)
		selem = skimage.morphology.disk(2)
		curr_mask[start[0], start[1]] = 1
		curr_mask = 1 - skimage.morphology.binary_dilation(
			curr_mask, selem) != True
		curr_mask = curr_mask ==1
		sem_map[curr_mask] = 3

		selem = skimage.morphology.disk(4)        
		goal_mat = 1 - skimage.morphology.binary_dilation(
			goal, selem) != True
		#goal_mat = goal
		goal_mask = goal_mat == 1
		sem_map[goal_mask] = 4


		#self.print_log(sem_map.shape, sem_map.min(), sem_map.max())
		#self.print_log(vis_mask.shape)
		#sem_map = self.compress_sem_map(sem_map)

		#color_palette = d3_40_colors_rgb.flatten()
		color_palette2 = [1.0, 1.0, 1.0,
				0.6, 0.6, 0.6,
				0.95, 0.95, 0.95,
				0.96, 0.36, 0.26,
				0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
				0.9400000000000001, 0.7818, 0.66,
				0.9400000000000001, 0.8868, 0.66,
				0.8882000000000001, 0.9400000000000001, 0.66,
				0.7832000000000001, 0.9400000000000001, 0.66,
				0.6782000000000001, 0.9400000000000001, 0.66,
				0.66, 0.9400000000000001, 0.7468000000000001,
				0.66, 0.9400000000000001, 0.9018000000000001,
				0.66, 0.9232, 0.9400000000000001,
				0.66, 0.8182, 0.9400000000000001,
				0.66, 0.7132, 0.9400000000000001,
				0.7117999999999999, 0.66, 0.9400000000000001,
				0.8168, 0.66, 0.9400000000000001,
				0.9218, 0.66, 0.9400000000000001,
				0.9400000000000001, 0.66, 0.9031999999999998,
				0.9400000000000001, 0.66, 0.748199999999999]
		
		color_palette2 += self.flattened.tolist()
		
		color_palette2 = [int(x*255.) for x in color_palette2]
		color_palette = color_palette2

		semantic_img = Image.new("P", (sem_map.shape[1],
									   sem_map.shape[0]))

		semantic_img.putpalette(color_palette)
		#semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
		semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
		semantic_img = semantic_img.convert("RGBA")

		semantic_img = np.flipud(semantic_img)
		self.semantic = semantic_img

		if self.args.visualize:
			cv2.imshow("Sem Map", self.semantic[:,:,[2,1,0,3]])
			cv2.waitKey(1)
		
		if self.args.save_pictures:
			cv2.imwrite(self.picture_folder_name + "Sem_Map/"+ "Sem_Map_" + str(self.steps_taken) + ".png", self.semantic[:,:,[2,1,0,3]])
			cv2.imwrite(self.picture_folder_name + "/Rgb/" + "Rgb_" + str(self.steps_taken) + ".png", self.img)

	def evaluate(self):
		goal_satisfied = self.get_goal_satisfied()
		if goal_satisfied:
			success = True
		else:
			success = False
			
		pcs = self.get_goal_conditions_met()
		goal_condition_success_rate = pcs[0] / float(pcs[1])

		# SPL
		path_len_weight = len(self.traj_data['plan']['low_actions'])
		s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(self.steps_taken))
		pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(self.steps_taken))

		# path length weighted SPL
		plw_s_spl = s_spl * path_len_weight
		plw_pc_spl = pc_spl * path_len_weight
		
		goal_instr = self.traj_data['turk_annotations']['anns'][self.r_idx]['task_desc']
		sliced = get_arguments(self.traj_data)[-1]
		
		# log success/fails
		log_entry = {'trial': self.traj_data['instance_id'],
					 #'scene_num': self.traj_data['scene']['scene_num'],
					 'type': self.traj_data['task_type'],
					 'repeat_idx': int(self.r_idx),
					 'goal_instr': goal_instr,
					 'completed_goal_conditions': int(pcs[0]),
					 'total_goal_conditions': int(pcs[1]),
					 'goal_condition_success': float(goal_condition_success_rate),
					 'success_spl': float(s_spl),
					 'path_len_weighted_success_spl': float(plw_s_spl),
					 'goal_condition_spl': float(pc_spl),
					 'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
					 'path_len_weight': int(path_len_weight),
					 'sliced':sliced,
					 'episode_no':  self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank,
					 'steps_taken': self.steps_taken}
					 #'reward': float(reward)}
		
		return log_entry, success
