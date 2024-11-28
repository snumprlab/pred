#Class that will replace thor_env_code and sem_exp_thor now
import math
import os, sys
import ast

import open_clip
from sentence_transformers import util

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
from models.segmentation.segmentation_helper import SemgnetationHelper
from models.segmentation.segmentation_helper import SemgnetationHelper
import utils.control_helper as  CH

from models.instructions_processed_LP.object_state import ObjectState
teach_dir = os.environ['TEACH_DIR']
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
sys.path.append(os.path.join(teach_dir, os.environ["FILM_model_dir"], 'models/instructions_processed_LP'))
# if torch.cuda.is_available():
# 	from models.instructions_processed_LP.get_arguments_from_bert import GetArguments
# else:
# 	from models.instructions_processed_LP.get_arguments_from_bert_cpu import GetArguments

from models.segmentation import alfworld_constants
import random

from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

from models.higt_to_low_action_generate import convert

large = alfworld_constants.STATIC_RECEPTACLES
large_objects2idx = {k:i for i, k in enumerate(large)}

small = alfworld_constants.OBJECTS_DETECTOR
small_objects2idx = {k:i for i, k in enumerate(small)}
INRECEP = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'LaundryHamper']
PICKUPABLE_OBJS = ['Towel', 'HandTowel', 'SoapBar', 'ToiletPaper', 'SoapBottle', 'Candle', 'ScrubBrush', 'Plunger', 'Cloth', 'SprayBottle', 'Book', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'CreditCard', 'AlarmClock', 'CD', 'Laptop', 'Watch',  'WateringCan', 'Newspaper', 'RemoteControl', 'Statue', 'BaseballBat', 'TennisRacket', 'Mug', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Fork', 'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 'ButterKnife', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'Bowl', 'Vase', 'TissueBox', 'Boots', 'PaperTowelRoll', 'Ladle', 'Kettle', 'GarbageBag', 'TeddyBear', 'Dumbbell', 'AluminumFoil']


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
		self.flattened = pickle.load(open(os.path.join(os.environ["FILM_model_dir"], "miscellaneous/flattened.p"), "rb"))
		
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
		self.pickup_executed = False
		#self.test_dict = read_test_dict(self.args.test, self.args.appended, 'unseen' in self.args.eval_split)
		self.test_dict =  None

		#Segmentation
		self.seg = SemgnetationHelper(self)

		# if self.args.use_bert:
		# 	self.ArgPredictor= GetArguments()

		#def force_cudnn_initialization():
		if torch.cuda.is_available():
			s = 32
			dev = torch.device('cuda')
			torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

		self.gpt_argument = False
		self.gpt_subgoal = True
		self.not_allowed_action = 0
		self.extracted_info = False
		self.gpt_task_type = True
		self.pass_obj = False
		self.current_gpt_subgoal = []
		with open('new_template/LLM_revised_actions.json') as f:
			self.gpt_revise_result = json.load(f)
		self.prev_point =-1

	def find_starting_horizon(self):
		actions = [ei['action_name']for ei in self.edh_instance['driver_action_history']]
		#count the number of lookups and lookdowns
		num_look_downs = sum([a=="Look Down" for a in actions])
		num_look_ups = sum([a=="Look Up" for a in actions])
		starting_hor = 30 + 30 *(num_look_downs - num_look_ups)
		return starting_hor


	def load_initial_scene(self, edh_instance, image):
		#############################
		self.interacted_obj_list = dict()
		self.prev_confirm_loc_obj = "NONE"
		#############################
		self.scene_pointer += 1
		self.img = image
		self.side_step_order = 0
		self.PREV_side_step_order = 0
		self.rotate_before_side_step_count = 0
		self.fails_cur = 0
		self.put_rgb_mask = None
		self.recep_mask = None  ##0430
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
		self.faucet_on_mask = None

		episode_no = self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank        
		self.episode_no = episode_no

		self.picking_wrong_num = 0     #####
		self.changed_plan_pointer =-1  ######
		self.check_picked_obj = False
		self.converted_subgoal = False
		self.pickup_executed =False
		self.picked_up_desired_obj = False
		self.pass_obj = False
		self.giveup_heat = False


		traj_data = edh_instance
		self.traj_data = traj_data
		self.r_idx = 0; r_idx = self.r_idx

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
		game_id = edh_instance['instance_id']
		
		if self.gpt_task_type:
			with open('new_template/predicted_task_type_tfd_edh.json') as f:
				gpt_task_type = json.load(f)
			task_type = gpt_task_type[game_id]

		if self.gpt_subgoal:
			with open('new_template/extracted_info_tfd_edh.json') as f:
				ext_info = json.load(f)
			obj_count = None
			if task_type == "Coffee":
				obj_target = "Mug"
				parent_target = None
	
			elif task_type == "Water Plant":
				Targets = list(ext_info[game_id].keys())
				Target = "X"; Parent = ""; Target_loc = ""
				try:
					for i in Targets:
						if i in ["Mug", "Cup", "Bowl"]:
							if "location" in ext_info[game_id][i]:
								if isinstance(ext_info[game_id][i]["location"], str):
									if ext_info[game_id][i]["location"] != "X" and ext_info[game_id][i]["location"] not in INRECEP:
										Target = i
										
								if isinstance(ext_info[game_id][i]["location"], list):
									for loc in ext_info[game_id][i]["location"]:
										if loc != "X" and loc not in INRECEP:
											Target = i
					if Target == "X":
						for i in Targets:
							if i in ["Mug", "Cup", "Bowl"]:
								if "location" in ext_info[game_id][i]:
									if isinstance(ext_info[game_id][i]["location"], str):
										if ext_info[game_id][i]["location"] != "X" and ext_info[game_id][i]["location"] in INRECEP:
											Target = i
											Target_loc = ext_info[game_id][i]["location"]
									if isinstance(ext_info[game_id][i]["location"], list):
										for loc in ext_info[game_id][i]["location"]:
											if loc != "X" and ext_info[game_id][i]["location"] in INRECEP:
												Target = i
												Target_loc = loc
				except: pass
				if Target == "X":
					Target = "Bowl"
				
				obj_target = Target
				parent_target = None

			elif task_type == "Boil X":
				obj_target = "Potato"
				parent_target = None
	
			elif task_type == "Plate Of Toast":
				obj_target = "Bread"
				parent_target = None
	
			elif task_type == "Clean All X":
				obj_target = "X"
				try:
					obj_target = list(ext_info[game_id].keys())[0]
				except: pass
				if obj_target not in PICKUPABLE_OBJS:
					obj_target = "Mug"
				parent_target = None
	
			elif task_type in ["N Cooked Slices Of X In Y", "N Slices Of X In Y"]:
				Targets = list(ext_info[game_id].keys())
				Target = "X"; Parent = "X"
				try:
					for i in Targets:
						if i in ["Tomato", "Lettuce", "Potato"]:
							Target = i
						if i in ["Bowl", "Plate"]:
							Parent = i
				except: pass
				try:					
					if Parent == "X":
						if "receptacle" in ext_info[game_id][Target]:
							if ext_info[game_id][Target]["receptacle"] != "X":
								Parent = ext_info[game_id][Target]["receptacle"]
				except: pass
				if Target == "X":
					Target = "Potato"
				if Parent == "X":
					Parent = "Plate"
				obj_target = Target
				parent_target = Parent

			elif task_type in ["Put All X In One Y", "Put All X On Y"] and game_id in ext_info:
				Targets = list(ext_info[game_id].keys())
				Target = "X"; Parent = "X"
				try:
					for i in Targets:
						if i in PICKUPABLE_OBJS:
							Target = i
				except: pass
				try:
					Parent = ext_info[game_id][Target]["receptacle"]
				except: pass
				if Target == "X": Target = 'Mug'
				if Parent == "X": Parent = 'CounterTop'
				obj_target = Target
				parent_target = Parent

			elif task_type == "Salad":
				obj_target = "Mug"
				parent_target = None
			elif task_type == "Sandwich":
				obj_target = "Mug"
				parent_target = None
			elif task_type == "Breakfast":
				obj_target = "Mug"
				parent_target = None




		if obj_target == "Bottle":
			obj_target = "Glassbottle"

		#If you ever get an object target or parent target not in , just replace them with "Mug"
		if not(obj_target in small_objects2idx) and not(obj_target in large_objects2idx):
			if not(obj_target  in ['Drinkware', 'Dishware', 'SportsEquipment', 'Condiments', 'Silverware', 'Cookware', 'Fruit', 'SmallHandheldObjects', 'Tableware'] ):
				obj_target = "Mug"

		if not(parent_target in small_objects2idx) and not(parent_target in large_objects2idx):
			if parent_target not in ['Tables', 'Chairs', 'Furniture']:
				parent_target = "Desk"

		real_obj_target = copy.deepcopy(obj_target); real_parent_target = copy.deepcopy(parent_target)
		if obj_target in ['Drinkware', 'Dishware', 'SportsEquipment', 'Condiments', 'Silverware', 'Cookware', 'Fruit', 'SmallHandheldObjects', 'Tableware']:
			if obj_target == 'Drinkware':
				obj_target = "Mug"
			elif obj_target == 'Dishware':
				obj_target = "Plate"
			elif obj_target == 'SportsEquipment':
				obj_target = 'BaseballBat'
			elif obj_target == 'Condiments':
				obj_target = 'SaltShaker'
			elif obj_target == "Silverware":
				obj_target = "Fork"
			elif obj_target == "Cookware":
				obj_target = 'Kettle'
			elif obj_target == 'Fruit':
				obj_target = 'Apple'
			elif obj_target =='Tableware':
				obj_target = "Plate"
			elif obj_target == 'SmallHandheldObjects':
				obj_target = "RemoteControl"


		if parent_target in ["Tables", "Chairs", "Furniture"]:
			if parent_target == "Tables":
				parent_target = "DiningTable"
			elif parent_target == "Chairs":
				parent_target = 'ArmChair'
			elif parent_target == "Furniture":
				parent_target = "Sofa"


		# gpt argument
		###################################################################################
		if self.gpt_argument:
			from alfred_utils.gen.constants import map_all_objects
			from alfred_utils.gen.constants import RECEPTACLES

			if obj_target not in map_all_objects:
				obj_target = random.choice(list(map_all_objects))
			if parent_target not in RECEPTACLES:
				parent_target = random.choice(list(RECEPTACLES))
		###################################################################################

		# 1016 gpt direct subgoal w/ sink->sinkbasin, waterplant intermediate obj 조정
		intermediate_obj_gpt = None
		if self.gpt_subgoal:
			### list of actions ###
			# gpt new template
			with open('new_template/subgoal_tfd_edh_0208_edh_revised.json') as f:
				gpt_subgoal = json.load(f)
			if game_id in gpt_subgoal:
				list_of_actions_gpt = []
				tmp = gpt_subgoal[game_id]
				self.current_gpt_subgoal = tmp
				for i in tmp:
					list_of_actions_gpt.append(i[:2])

				if task_type == "Water Plant" :
					# change intermediate_obj
					for i in list_of_actions_gpt:
						if i[0] in ['Mug', 'Cup', 'Bowl']:
							intermediate_obj_gpt = i[0]
							break


		prev_actions = edh_instance['driver_action_history']
		obj_state = ObjectState(edh_instance['dialog_history'], prev_actions, task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target, intermediate_obj_gpt = intermediate_obj_gpt)
		sliced = obj_state.slice_needs_to_happen
		#####################
		list_of_actions = obj_state.future_list_of_highlevel_actions 
		categories_in_inst = obj_state.categories_in_inst
		second_object = obj_state.second_object
		caution_pointers = obj_state.caution_pointers

		if self.args.add_faucet_toggleoff:
			if task_type in ['Coffee', 'Clean All X']:
				list_of_actions = [('Faucet', 'ToggleObjectOff')] + list_of_actions
				if not ('Faucet' in categories_in_inst):
					categories_in_inst = ['Faucet'] + categories_in_inst
				caution_pointers = [i+1 for i in caution_pointers]
				#Ignore second object for now
    
		# gpt subgoal (direct)
		###################################################################################
		if self.gpt_subgoal:
			with open('FILM_model/classes.txt', 'r') as file:
				content = file.read().strip()  # 파일 내용 읽기 및 공백 제거
				word_list = content.split(', ')			
			### list of actions ###
   
			# gpt new template
			with open('new_template/subgoal_tfd_edh_0208_edh_revised.json') as f:
				gpt_subgoal = json.load(f)
    

			if game_id in gpt_subgoal:
				list_of_actions = []
				for action in gpt_subgoal[game_id]:
					list_of_actions.append([action[0], action[1],action[2][0],action[3]])

				# invalid action name check
				for i in list_of_actions:
					if i[1] not in ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "PourObject", "SliceObject"] \
         			or i[0] not in word_list:
						list_of_actions = [[obj_target, 'PickupObject',i[2],i[3]]]
						self.not_allowed_action += 1
						naa = open("not_allowed_action_count.txt", "a+")
						naa.write(str(self.not_allowed_action)+": "+ str(i)+"\n")
						naa.close()
				for i in list_of_actions:
					if i[0] == 'Sink':
						i[0] = 'SinkBasin'

			else:
				list_of_actions = [[obj_target, 'PickupObject','None',0]]
	
		####################################################################################################################3
		if self.gpt_subgoal or self.extracted_info:
			### categories_in_inst ###
			from collections import OrderedDict
			categories_in_inst = list(OrderedDict({k[0]: 1 for k in list_of_actions}).keys())

			### caution pointers  && second object ###
			caution_pointers = []
			second_object = []
			idx_ckecked = []
			for idx, action in enumerate(list_of_actions):
				second_object.append(None)
				if action[1] == 'PutObject':
					caution_pointers.append(idx)
				elif action[1] == 'ToggleObjectOn':
					caution_pointers.append(idx)
				elif action[1] == 'ToggleObjectOff':
					caution_pointers.append(idx)
				elif action[1] == 'OpenObject':
					caution_pointers.append(idx)
				elif action[1] == 'CloseObject':
					caution_pointers.append(idx)
				elif action[1] == 'PourObject':
					caution_pointers.append(idx)
     
				if task_type in ['Put All X In One Y', 'Put All X On Y']: #,'Clean All X'] and :
					if action[1] == 'PickupObject' and action[2]=="Move" and idx not in idx_ckecked and len(list_of_actions)>1:
						if list_of_actions[idx+1][1] == "PutObject" :
							second_object[idx] = list_of_actions[idx+1][0]
						elif list_of_actions[idx+2][1] == "PutObject" :
							second_object[idx] = list_of_actions[idx+2][0]
					



		self.first_goal_sliced_state = False
		for o in obj_state.init_obj_state_dict:
			if obj_state.init_obj_state_dict[o]["Sliced"]:
				self.first_goal_sliced_state = True

		#Save these too

				
  
  
		
		if self.args.save_pictures:
			pickle.dump(edh_instance, open(self.picture_folder_name +"/edh_dir/" + str(self.steps_taken) +'_edh_instance.p', 'wb')) 
			pickle.dump(obj_state, open(self.picture_folder_name +"/edh_dir/" + str(self.steps_taken) +'_obj_state.p', 'wb')) 

		self.print_log("Dialog History ", edh_instance['dialog_history'])

		self.print_log("Task type is ", task_type)
		self.print_log("Unmatched states are ", obj_state.unmatched_states)
		self.print_log("List of actions ", list_of_actions)
		self.print_log("categories_in_inst ", categories_in_inst)
		self.print_log("Edh ID: ",  edh_instance['instance_id']); print("Edh ID: ",  edh_instance['instance_id'])

		print("")
		print("-----------------   Dialog History   --------------------------")
		for idx in range(len(edh_instance['dialog_history'])):
			print( edh_instance['dialog_history'][idx])
		print("")
		print("-----------------   Task Type && List of actions   --------------------------")
		print("---- Task type ")
		print(task_type)
		print("")
		print("---- Unmatched states objs ")
		for idx in list(obj_state.unmatched_states.items()): 
			print(idx)		
		print("")
		print("---- List of actions ")

		for idx in range(len(list_of_actions)):
			print(str(idx) +". "+ str(list_of_actions[idx]))
		print("")
		print("Caution Pointers : ", caution_pointers)
		print("Second Object : ", second_object)



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
		print("----------------------------------------------------------------------------")
		print("----------------------------------------------------------------------------")
		#Now we will do set up scene after the three look down's

		starting_hor = self.find_starting_horizon()
		obs, info = super().setup_scene(starting_hor)
		#print("SETUP SCENE END")
		goal_name = list_of_actions[0][0]
		print("RESET GOAL START")
		info, return_actions = self.reset_goal(True, goal_name, None)
		print("RESET GOAL END")


		if task_type == "Water Plant" :
			for obj in obj_state.FILLABLE_OBJS:
				if obj_state.intermediate_obj in self.total_cat2idx:
					self.total_cat2idx[obj] = self.total_cat2idx[obj_state.intermediate_obj]
					self.cat_equate_dict[obj] = obj_state.intermediate_obj
######################################################################################     
		if task_type == "Boil X" :
			for obj in obj_state.FILLABLE_OBJS:
				if "Bowl" in self.total_cat2idx:
					self.total_cat2idx[obj] = self.total_cat2idx["Bowl"]
					self.cat_equate_dict[obj] = "Bowl"
########################################################################################
		if obj_target in ['Drinkware', 'Dishware', 'SportsEquipment', 'Condiments', 'Silverware', 'Cookware', 'Fruit', 'SmallHandheldObjects', 'Tableware']:
			if real_obj_target == 'Drinkware':
				for obj in obj_state.Drinkware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'Dishware':
				for obj in obj_state.Dishware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'SportsEquipment':
				for obj in obj_state.SportsEquipment:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'Condiments':
				for obj in obj_state.Condiments:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == "Silverware":
				for obj in obj_state.Silverware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'Cookware':
				for obj in obj_state.Cookware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target
			elif real_obj_target == 'Fruit':
				for obj in obj_state.Fruit:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target

			elif real_obj_target == 'SmallHandheldObjects':
				for obj in obj_state.SmallHandheldObjects:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target

			elif real_obj_target =='Tableware':
				for obj in obj_state.Tableware:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[obj_target]
						self.cat_equate_dict[obj] = obj_target

		if real_parent_target in ["Tables", "Chairs", 'Furniture']:
			if real_parent_target == "Tables":
				for obj in obj_state.Tables:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[parent_target]
						self.cat_equate_dict[obj] = parent_target

			if real_parent_target == "Chairs":
				for obj in obj_state.Chairs:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[parent_target]
						self.cat_equate_dict[obj] = parent_target

			if real_parent_target == "Furniture":
				for obj in obj_state.Furniture:
					if obj_target in self.total_cat2idx:
						self.total_cat2idx[obj] = self.total_cat2idx[parent_target]
						self.cat_equate_dict[obj] = parent_target

		if sliced:
			self.total_cat2idx['ButterKnife'] = self.total_cat2idx['Knife'] 
			self.cat_equate_dict['ButterKnife'] = 'Knife' #if DeskLamp is found, consider it as FloorLamp
			
		
		actions_dict = {'task_type': task_type, 'list_of_actions': list_of_actions, 'second_object': second_object, 'total_cat2idx': self.total_cat2idx, 'sliced':self.sliced}
		self.print_log('total cat2idx is ', self.total_cat2idx)
		
		self.actions_dict = actions_dict


		self.seg.update_agent(self)
		#self.consecutive_interaction_executed = False
		self.last_action_ogn = None
		print("Ran this till the end")
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

#######################################################
	def _get_approximate_success(self, prev_rgb, frame, action):
		prev_rgb = copy.deepcopy(prev_rgb)
		#prev_rgb[250:, :] = 0
		frame = copy.deepcopy(frame)
		#frame[250:, :] = 0
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
				if max_area_below100 > 15:
					success=True
			else:
				if not(action in ['LookDown_0', 'LookUp_0']):
					self.print_log("Interaction not in category in _get_approximate_success ", action)
					raise Exception("INTERACTION not in category in _get_approximate_success : " + str(action))
			self.print_log("Approx Success is ", success)
		return success
###################################################################

	def update_interacted_obj(self) :
		# update object the interacted by agent more at least once
		interaction_actions = set(["ToggleObjectOn", "PourObject", "ToggleObjectOff", "OpenObject","CloseObject", 'SliceObject', "PickupObject", "PutObject"]) 


		if self.last_success and self.prev_action in interaction_actions and not self.converted_subgoal:
			if self.prev_action in ['PickupObject'] :
				if  self.current_subgoal[0] in self.interacted_obj_list:
					already_sliced = False
					for obj_action in  self.lislist_of_actions:
						if obj_action[0]== self.current_subgoal[0] and obj_action[1]==  "SliceObject":
							already_sliced= True
							break
					if not already_sliced :
						del self.interacted_obj_list[self.current_subgoal[0]]
						# print(self.interacted_obj_list)

					
			else:  ##toggle을 뺄까?? mask는,,, 바뀐 부분만으로 업데이트 해야한다 
				self.interacted_obj_list[self.current_subgoal[0]] =dict(waypoint = self.cur_start, mask=  self.interaction_mask)
				# print(print(self.interacted_obj_list))

		
				

		if self.converted_subgoal and self.prev_action in ["PutObject"]:
			self.converted_subgoal = False
   
	#Everything up to va_interact_new
	def semexp_step_internal_right_after_sim_step(self, last_success_special_treatment=None, planner_inputs=None,infos=None,local_map=None):
		#First call thor_step_internal_right_after_sim_step
		state, info = self.thor_step_internal_right_after_sim_step()
		self.obs_step = state; self.info = info

		#First thing after va_interact_new (the most outer thing in thor)
		#success = CH._get_approximate_success(self.prev_rgb, self.img, action)
		if last_success_special_treatment is None:
			success= self._get_approximate_success(self.prev_rgb, self.img, self.prev_action)
			self.last_success = success


			if  self.prev_action == "PickupObject" and success:
				self.pickup_executed =True 
			elif self.prev_action in ["ToggleObjectOn", "PourObject", "ToggleObjectOff", "OpenObject","CloseObject", 'SliceObject', "PutObject"] and success:
				self.pickup_executed =False 

			if self.prev_action == "PutObject" and not success:
				
				########################  Give up clean ################# 
				if  planner_inputs['list_of_actions'][self.pointer][2] == "Clean" and planner_inputs['list_of_actions'][self.pointer][0] =="SinkBasin":
					if self.task_type == "Coffee" :
						del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
						del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
						del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet
						del planner_inputs['list_of_actions'][self.pointer] # pickup Mug

						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						del self.second_object[self.pointer]

					#####  Pour 여부 판단??
					if self.pointer < len(planner_inputs['list_of_actions']) and planner_inputs['list_of_actions'][self.pointer][2] == "Pour" :
         
						mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only('Mug')

						if mask is not None  and np.sum(mask) >1 :
							test_img =   mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
							test_img = self.crop_img_fitting_on_obj(test_img)
							test_img = self.enhance_image_brightness_and_sharpness(test_img,127)
         
							largest_lab = self.Check_similarity(test_img, "Fill", obj_name='Mug')
							if largest_lab  == 0 :
								del planner_inputs['list_of_actions'][self.pointer]
								del self.second_object[self.pointer]
							
       
       
					else : 
						del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
						del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
						del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet

						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						del self.second_object[self.pointer]

						## 무조건 내려놓기 
						recep = 'CounterTop'
						for recep_option in ['CounterTop','Desk','Sofa']:
							cn = self.total_cat2idx[recep_option]
							if local_map[cn+4,:,:].sum() > 0. :
								recep = recep_option
								break
						planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
						self.second_object.insert(self.pointer,None)



					#### 마지막 action의 경우 ,,, fail 방지용 ,,, Put?
					if self.pointer == (len(planner_inputs['list_of_actions'])) :
						recep = 'CounterTop'
						for recep_option in ['CounterTop','Desk','Sofa']:
							cn = self.total_cat2idx[recep_option]
							if local_map[cn+4,:,:].sum() > 0. :
								recep = recep_option
								break
						planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
						self.second_object.insert(self.pointer,None)
							
					self.set_caution_point(planner_inputs,infos)	        
		###################################################################
			########################  Give up fill  ######################
				elif  planner_inputs['list_of_actions'][self.pointer][2] == "Fill" and planner_inputs['list_of_actions'][self.pointer][0] =="SinkBasin":
						del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
						del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
						del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet
						del planner_inputs['list_of_actions'][self.pointer]	# Pickup 

						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						del self.second_object[self.pointer]
						
						self.set_caution_point(planner_inputs,infos)   
			######################  Give up heat  ######################
				elif  planner_inputs['list_of_actions'][self.pointer][2] == "Heat" and planner_inputs['list_of_actions'][self.pointer][0] =="Microwave":
        
						planner_inputs['list_of_actions'].insert(self.pointer,planner_inputs['list_of_actions'][self.pointer+1]) # close Microwave
						self.second_object.insert(self.pointer,None)
      
						del planner_inputs['list_of_actions'][self.pointer+1] # Put Microwave
						del planner_inputs['list_of_actions'][self.pointer+1] # close Microwave
						del planner_inputs['list_of_actions'][self.pointer+1] # toggleon Microwave 
						del planner_inputs['list_of_actions'][self.pointer+1] # toggleff Microwave 
						del planner_inputs['list_of_actions'][self.pointer+1] # open Microwave

						del self.second_object[self.pointer+1]
						del self.second_object[self.pointer+1]
						del self.second_object[self.pointer+1]
						del self.second_object[self.pointer+1]
						del self.second_object[self.pointer+1]
						
						self.set_caution_point(planner_inputs,infos)  
						self.giveup_heat =True



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
		self.update_interacted_obj()
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


	def crop_img_fitting_on_obj(self, img):
			img_gray=cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
			contours, _ = cv2.findContours(img_gray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			contours_xy = np.array(contours)
			contours_xy.shape
			x_min, x_max = 0,0
			value = list()
			for i in range(len(contours_xy)):
				for j in range(len(contours_xy[i])):
					value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
					x_min = min(value)
					x_max = max(value)
			
			# y의 min과 max 찾기
			y_min, y_max = 0,0
			value = list()
			for i in range(len(contours_xy)):
				for j in range(len(contours_xy[i])):
					value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
					y_min = min(value)
					y_max = max(value)

			x = x_min
			y = y_min
			w = x_max-x_min
			h = y_max-y_min
			img = img[y:y+h, x:x+w].astype(np.uint8)
   
			return img


	def enhance_image_brightness_and_sharpness(self,test_img,brightness_value): 
		###### Brightness adjustment
		img_trim = cv2.cvtColor(test_img, cv2.COLOR_RGB2YUV)

		diferrences = brightness_value - int(np.mean(img_trim[:,:,0][np.where(img_trim[:,:,0] > 0)]))
		if diferrences < 0 :
			img_trim[:,:,0][np.where(img_trim[:,:,0] > diferrences*(-1))]  = img_trim[:,:,0][np.where(img_trim[:,:,0] > diferrences*(-1))] + diferrences
		else : 
			img_trim[:,:,0][np.where(img_trim[:,:,0] < 255-diferrences)]   = img_trim[:,:,0][np.where(img_trim[:,:,0] < 255-diferrences)]    + diferrences
		test_img = cv2.cvtColor(img_trim, cv2.COLOR_YUV2BGR)

		#### sharpening image 
		kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
		test_img = cv2.filter2D(test_img,-1,kernel_sharpening )
  
		return test_img



	def Check_similarity(self,test_img, state,obj_name=None):

		device = "cuda" if torch.cuda.is_available() else "cpu"
		# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
		model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained="laion400m_e32")
		model.to(device)
		def imageEncoder(img):
			img1 = Image.fromarray(img).convert('RGB')
			img1 = preprocess(img1).unsqueeze(0).to(device)
			img1 = model.encode_image(img1)
			return img1
		def generateScore(image1, image2):
			img1 = imageEncoder(image1)
			img2 = imageEncoder(image2)
			cos_scores = util.pytorch_cos_sim(img1, img2)
			score = round(float(cos_scores[0][0])*100, 2)
			return score

		img_dir=[]
		indx_detail =[] 
		largest_lab = None
  
		if  obj_name =="SinkBasin" :
			indx_detail.append("SinkBasin / On ")
			img_dir.append("atrribute_imgs/SinkBain_toggledon_1.jpeg")

			indx_detail.append("SinkBasin / Off ")
			img_dir.append("atrribute_imgs/SinkBain_toggledoff_1.jpeg")   

		elif obj_name =="Microwave" :
			indx_detail.append("Microwave / On ")
			img_dir.append("atrribute_imgs/Microwave_toggledon_1.jpeg")

			indx_detail.append("Microwave / Off ")
			img_dir.append("atrribute_imgs/Microwave_toggledoff_1.jpeg")   
   
		elif state == "Fill" :
			if obj_name in ['Cup','Mug','Bowl'] :
			# if obj_name == 'Cup' :
				indx_detail.append("Cup / Empty ")
				img_dir.append("atrribute_imgs/Cup_empty_clean_1.jpeg")

				indx_detail.append("Cup / Water ")
				img_dir.append("atrribute_imgs/Cup_water_clean_1.jpeg")   
			# elif obj_name == 'Mug' :
				indx_detail.append("Mug / Empty ")
				img_dir.append("atrribute_imgs/Mug_empty_clean_0.jpeg")

				indx_detail.append("Mug / Water ")
				img_dir.append("atrribute_imgs/Mug_water_clean_0.jpeg")   
			# elif obj_name == 'Bowl' :
				indx_detail.append("Bowl / Empty ")
				img_dir.append("atrribute_imgs/Bowl_empty_clean_0.jpeg")

				indx_detail.append("Bowl / Water ")
				img_dir.append("atrribute_imgs/Bowl_water_clean_0.jpeg")   
			else :
				largest_lab = 1 
    
    
		elif state == "Clean" :
			# if obj_name == 'Cup' :
			# 	indx_detail.append("Cup / Dirty ")
			# 	img_dir.append("atrribute_imgs/Cup_empty_dirty_1.jpeg")
    
			# 	indx_detail.append("Cup / Clean ")
			# 	img_dir.append("atrribute_imgs/Cup_empty_clean_1.jpeg")  
    
			# if obj_name == 'Mug' :
			# 	indx_detail.append("Mug / Dirty ")
			# 	img_dir.append("atrribute_imgs/Mug_empty_dirty_1.jpeg")

			# 	indx_detail.append("Mug / Clean ")
			# 	img_dir.append("atrribute_imgs/Mug_empty_clean_1.jpeg")   
    
			# 	indx_detail.append("Mug / Dirty ")
			# 	img_dir.append("atrribute_imgs/Mug_empty_dirty_0.jpeg")

			# 	indx_detail.append("Mug / Clean ")
			# 	img_dir.append("atrribute_imgs/Mug_empty_clean_0.jpeg")   
    
			if obj_name == 'Bowl' :
				indx_detail.append("Bowl / Dirty ")
				img_dir.append("atrribute_imgs/Bowl_empty_dirty_0.jpeg")

				indx_detail.append("Bowl / Clean ")
				img_dir.append("atrribute_imgs/Bowl_empty_clean_0.jpeg")   
    
			elif obj_name == 'Plate' :
				indx_detail.append("Plate / Dirty ")
				img_dir.append("atrribute_imgs/Plate_dirty_1.jpeg")

				indx_detail.append("Plate / Clean ")
				img_dir.append("atrribute_imgs/Plate_clean_1.jpeg")  
    
			# elif obj_name == 'Cloth' :
			# 	indx_detail.append("Cloth / Dirty ")
			# 	img_dir.append("atrribute_imgs/Cloth_dirty_0.jpeg")

			# 	indx_detail.append("Cloth / Clean ")
			# 	img_dir.append("atrribute_imgs/Cloth_clean_0.jpeg")   
    
			# elif obj_name == 'Pot' :
			# 	indx_detail.append("Pot / Dirty ")
			# 	img_dir.append("atrribute_imgs/Pot_empty_dirty_3.jpeg")

			# 	indx_detail.append("Pot / Clean ")
			# 	img_dir.append("atrribute_imgs/Pot_empty_clean_3.jpeg")   
    
			# elif obj_name == 'Pan' :
			# 	indx_detail.append("Pan / Clean ")
			# 	img_dir.append("atrribute_imgs/Pan_dirty_1.jpeg")

			# 	indx_detail.append("Pan / Water ")
			# 	img_dir.append("atrribute_imgs/Pan_clean_1.jpeg")   
    
			else :
				largest_lab = 0
       


		if largest_lab is None:
			largest_similarity_score=-1
			for idx in range(len(img_dir)):
				obj_only      =cv2.imread( img_dir[idx])
				print(f"similarity Score: ",indx_detail[idx], round(generateScore(test_img.astype(np.uint8), obj_only.astype(np.uint8)), 2))
				if  largest_similarity_score < round(generateScore(test_img.astype(np.uint8), obj_only.astype(np.uint8)), 2) :
					largest_similarity_score = round(generateScore(test_img.astype(np.uint8), obj_only.astype(np.uint8)), 2)
					largest_lab = idx
			print()
			print(largest_lab,"     :    " ,indx_detail[largest_lab])		
			print()
  
		else :
			print()
			print("Keep Fallow current subgoal  : Pick up undesired obj ")		
			print()      
  
		
		return int(largest_lab%2)


	def set_caution_point(self,planner_inputs,infos) :
		new_caution=[]
   
		for idx, action in enumerate(planner_inputs['list_of_actions']):
			if action[1] == 'PutObject':
				new_caution.append(idx)
			elif action[1] == 'ToggleObjectOn':
				new_caution.append(idx)
			elif action[1] == 'ToggleObjectOff':
				new_caution.append(idx)
			elif action[1] == 'OpenObject':
				new_caution.append(idx)
			elif action[1] == 'CloseObject':
				new_caution.append(idx)
			elif action[1] == 'PourObject':
				new_caution.append(idx)
    
		self.caution_pointers= new_caution
		self.goal_idx = self.total_cat2idx[planner_inputs['list_of_actions'][self.pointer][0]]
		self.info['goal_cat_id']=self.goal_idx
		self.info['goal_name'] = planner_inputs['list_of_actions'][self.pointer][0]
		self.changed_plan_pointer = self.pointer
		print("------------------NEW PLAN-----------------")
		for id in range(len(planner_inputs['list_of_actions'])) :
			print(planner_inputs['list_of_actions'][id])
		print("---------------------------------------------")
		print(self.caution_pointers)
		print(self.second_object)


	def update_and_refine_actions_by_GPT(self,current_gpt_subgoal,current_idx):


		consider_action =[]
		consider_idx = []
		for current_gpt_subgoal_ in current_gpt_subgoal:
			if current_gpt_subgoal_[3]>= current_idx and current_gpt_subgoal_[3] not in consider_idx:
				consider_action.append(current_gpt_subgoal_[2])
				consider_idx.append(current_gpt_subgoal_[3])
		consider_action[0][1] = 'Target'
		# print(consider_action)


		if str(consider_action) in self.gpt_revise_result :
			revised_result = self.gpt_revise_result[str(consider_action)]
		# print(revised_result)


		erased_action = str(consider_action)[1:-1].replace(revised_result[1:-1], '')
		if erased_action[-2:]==', ': erased_action= erased_action[:-2]

		print("Delete action : ", erased_action)
		return erased_action, revised_result

	def update_and_refine_actions_by_GPT_low(self,left_acion_num,planner_inputs,method=None):

		consider_action =[]
		for idx in range(self.pointer, self.pointer+left_acion_num) :
			consider_action.append([planner_inputs[idx][0],planner_inputs[idx][1]])
   
		if method=="Observation" : ## erase 
			consider_action[0][0] = 'Recep'
			consider_action[1][0] = 'Target'
			consider_action[2][0] = 'Recep'

			if str(consider_action) in self.gpt_revise_result :
				revised_result = self.gpt_revise_result[str(consider_action)]
    
			# erased_action = str(consider_action)[1:-1].replace(revised_result[1:-1], '')
			for revised_element in ast.literal_eval(revised_result):
				if revised_element in consider_action:
					consider_action.remove(revised_element)
   
   
			erased_action = str(consider_action[0:2])
			print("Erased action : ", erased_action)


		elif method=="Double_check" :  ## add

			consider_action =[]
			for idx in range(self.pointer-1, self.pointer+left_acion_num) :
				consider_action.append([planner_inputs[idx][0],planner_inputs[idx][1]])
    
			consider_action[0][0] = 'Target'

			if str(consider_action) in self.gpt_revise_result :
				revised_result = self.gpt_revise_result[str(consider_action)]
    
			revised_result = ast.literal_eval(revised_result)
			for consider_item in consider_action:
				if consider_item in revised_result:
					revised_result.remove(consider_item)
			erased_action = str(revised_result[0:2])
			print("Added action : ", erased_action)

		else : ## attribute(toggle..)
			if str(consider_action) in self.gpt_revise_result :
				revised_result = self.gpt_revise_result[str(consider_action)]

			# print(revised_result)
			erased_action = str(revised_result)[1:-1].replace(str(consider_action)[1:-1], '')
			if erased_action[-2:]==', ': erased_action= erased_action[:-2]
			print("Added action : ", erased_action)


		return erased_action, revised_result
  
  
	def game_over(self,left_acion_num,planner_inputs,local_map,infos) :

		fname = 'results/exception_files/' +"gam_over_error"+ '.json'
		if not os.path.isfile(fname) :
			users = {"ErrorID" : [self.edh_instance['instance_id']] }
			with open(fname, 'w') as f:
				json.dump(users, f)
		else :
			data = open(fname, 'r').read()
			data = json.loads(data)
			error_list = data['ErrorID']
			error_list.append(self.edh_instance['instance_id'])
			users = {"ErrorID" : error_list}
			with open(fname, 'w') as f:
				json.dump(data, f)






		for _ in range(left_acion_num) :
			del planner_inputs['list_of_actions'][self.pointer]
			del self.second_object[self.pointer]

		recep = 'CounterTop'
		for recep_option in ['CounterTop','Desk','Sofa']:
			cn = self.total_cat2idx[recep_option]
			if local_map[cn+4,:,:].sum() > 0. :
				recep = recep_option
				break 
		# 집고있는거 내려놓고 game over 
		planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
		self.second_object.insert(self.pointer,None)

		self.set_caution_point(planner_inputs,infos)




	def  revise_all_following_actions(self,left_acion_num, planner_inputs,new_lowactions,local_map,infos) :

		for _ in range(left_acion_num) :
			del planner_inputs['list_of_actions'][self.pointer]
			del self.second_object[self.pointer]
   
		for new_lowaction in new_lowactions:
			planner_inputs['list_of_actions'].append(new_lowaction)
			self.second_object.append(None)
		for idx, action in enumerate(planner_inputs['list_of_actions']):
			if self.task_type in ['Put All X In One Y', 'Put All X On Y']: #,'Clean All X'] and :
				if action[1] == 'PickupObject' and action[2]=="Move" and len(planner_inputs['list_of_actions'])>1:
					if planner_inputs['list_of_actions'][idx+1][1] == "PutObject" :
						self.second_object[idx] = self.list_of_actions[idx+1][0]
					elif planner_inputs['list_of_actions'][idx+2][1] == "PutObject" :
						self.second_object[idx] = self.list_of_actions[idx+2][0]
      
		self.set_caution_point(planner_inputs,infos)
  
  
  
	def  revise_same_highlevel_following_actions(self,left_acion_num, planner_inputs,new_lowactions,local_map,infos) :

		for _ in range(left_acion_num) :
			del planner_inputs['list_of_actions'][self.pointer]
			del self.second_object[self.pointer]
   
		for new_lowaction in reversed(new_lowactions):
			planner_inputs['list_of_actions'].insert(self.pointer,new_lowaction)
			self.second_object.insert(self.pointer,None)
		for idx, action in enumerate(planner_inputs['list_of_actions']):
			if self.task_type in ['Put All X In One Y', 'Put All X On Y']: #,'Clean All X'] and :
				if action[1] == 'PickupObject' and action[2]=="Move" and len(planner_inputs['list_of_actions'])>1:
					if planner_inputs['list_of_actions'][idx+1][1] == "PutObject" :
						self.second_object[idx] = self.list_of_actions[idx+1][0]
					elif planner_inputs['list_of_actions'][idx+2][1] == "PutObject" :
						self.second_object[idx] = self.list_of_actions[idx+2][0]
      
		self.set_caution_point(planner_inputs,infos)
  
  
  
  


	def semexp_plan_act_and_preprocess_revise_plan(self, planner_inputs, goal_spotted,infos,local_map,simulator):
		#These are things from plan_act_and_preprocess
  
##############################################################################  
		simulator.controller.step(action="Pass", agentId=0, renderObjectImage=True)

  
  
        ###################visualize################################      
        # instance_segs = np.array(simulator.controller.last_event.instance_segmentation_frame)
		# color_to_object_id = simulator.controller.last_event.color_to_object_id  
		# cv2.imshow('current_imag', instance_segs.astype(np.uint8))
		# cv2.waitKey(1)
		# point_img= instance_segs.copy()
		# point_img[pixel_y,pixel_x,:]= 255
		# cv2.imshow('Point', point_img)
        ##################################################
#################################################################################  


		self.pointer = planner_inputs['list_of_actions_pointer']

		# self.steps += 1
		# self._visualize(planner_inputs)

		self.Return_actions = []
		

		if  self.pickup_executed and  planner_inputs['consecutive_interaction'] == None :
			print("---- Check  Picked Up Object ---------")
			self.pickup_executed = False
			self.check_picked_obj = True

			self.Return_actions += self.set_back_to_angle(0)
			self.steps += len(self.Return_actions )
			
			return self.Return_actions
################# Attribute : FILL , Attribute : Clean  #############
		elif self.picked_up_desired_obj:  #Check attribute
			self.picked_up_desired_obj = False 
			# mask = self.seg.sem_seg_get_instance_mask_from_obj_type(planner_inputs['list_of_actions'][self.pointer-1][0])
			mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(planner_inputs['list_of_actions'][self.pointer-1][0])

			if mask is not None  and np.sum(mask) >1 :
				# cv2.imshow("tmp",self.img[:,:,::-1])
				# cv2.waitKey()
				test_img =   mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
				test_img = self.crop_img_fitting_on_obj(test_img)
				test_img = self.enhance_image_brightness_and_sharpness(test_img,127)
        
				if  planner_inputs['list_of_actions'][self.pointer][2] == "Fill":
					largest_lab = self.Check_similarity(test_img, "Fill", obj_name=planner_inputs['list_of_actions'][self.pointer-1][0])
					if largest_lab  == 1 :
						## Load GPT revised action 
						try : 
							erased_action, revised_result = self.update_and_refine_actions_by_GPT(self.current_gpt_subgoal,current_idx = planner_inputs['list_of_actions'][self.pointer][3])
							if erased_action == "['Fill', 'Target', 'None']" :
								del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
								del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
								del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet
								del planner_inputs['list_of_actions'][self.pointer]	# Pickup 
			
								del self.second_object[self.pointer]
								del self.second_object[self.pointer]
								del self.second_object[self.pointer]
								del self.second_object[self.pointer]
								
								self.set_caution_point(planner_inputs,infos)
							else : 
								new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
								left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
			
								self.revise_all_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)
						# if fail to load GPT revised action 
						except : 
							left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
							self.game_over(left_acion_num, planner_inputs,local_map,infos)


							
      			
				elif  planner_inputs['list_of_actions'][self.pointer][2] == "Clean":
					largest_lab = self.Check_similarity(test_img, "Clean", obj_name=planner_inputs['list_of_actions'][self.pointer-1][0])
					if largest_lab == 1 :
						## Load GPT revised action 
						try : 
							erased_action, revised_result = self.update_and_refine_actions_by_GPT(self.current_gpt_subgoal,current_idx = planner_inputs['list_of_actions'][self.pointer][3])
							if erased_action == "['Clean', 'Target', 'None']" :
								if planner_inputs['list_of_actions'][self.pointer][1] == "CloseObject" :
									del planner_inputs['list_of_actions'][self.pointer] # CloseObject
									del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
									del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
									del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet
			
									del self.second_object[self.pointer]
									del self.second_object[self.pointer]
									del self.second_object[self.pointer]
									del self.second_object[self.pointer]
								else :
									del planner_inputs['list_of_actions'][self.pointer] # Put sinkbasin
									del planner_inputs['list_of_actions'][self.pointer] # Toggleon Faucet
									del planner_inputs['list_of_actions'][self.pointer] # Toggleoff Faucet
			
									del self.second_object[self.pointer]
									del self.second_object[self.pointer]
									del self.second_object[self.pointer]
									

								if  self.pointer < len(planner_inputs['list_of_actions']) \
									and planner_inputs['list_of_actions'][self.pointer][0] == planner_inputs['list_of_actions'][self.pointer-1][0] and \
									planner_inputs['list_of_actions'][self.pointer][1] == planner_inputs['list_of_actions'][self.pointer-1][1] == "PickupObject" :
										del planner_inputs['list_of_actions'][self.pointer] # 같은걸 다시 집으려고할 때 action  지운다
				
										del self.second_object[self.pointer]
			
								elif self.pointer < len(planner_inputs['list_of_actions']) \
									and planner_inputs['list_of_actions'][self.pointer][0] != planner_inputs['list_of_actions'][self.pointer-1][0] and \
									planner_inputs['list_of_actions'][self.pointer-1][1]  == "PickupObject" and planner_inputs['list_of_actions'][self.pointer][1] in ["PickupObject","OpenObject"]:
										recep = 'CounterTop'
										for recep_option in ['CounterTop','Desk','Sofa']:
											cn = self.total_cat2idx[recep_option]
											if local_map[cn+4,:,:].sum() > 0. :
												recep = recep_option
												break 
										# 이미 집고 있는데 다른걸 집으려고 하면 내려놓고 집는다.
										planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
										self.second_object.insert(self.pointer,None)



								if self.pointer < len(planner_inputs['list_of_actions']) and planner_inputs['list_of_actions'][self.pointer][2] == "Pour" :
									largest_lab = self.Check_similarity(test_img, "Fill", obj_name=planner_inputs['list_of_actions'][self.pointer-1][0])
									if largest_lab  == 0 :
										del planner_inputs['list_of_actions'][self.pointer]
										del self.second_object[self.pointer]
										
								if self.pointer == (len(planner_inputs['list_of_actions'])) :
									recep = 'CounterTop'
									for recep_option in ['CounterTop','Desk','Sofa']:
										cn = self.total_cat2idx[recep_option]
										if local_map[cn+4,:,:].sum() > 0. :
											recep = recep_option
											break
									planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
									self.second_object.insert(self.pointer,None)
										
								self.set_caution_point(planner_inputs,infos)		
							else : 
								new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
								left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
			
								self.revise_all_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)
						# if fail to load GPT revised action 
						except : 
							left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
							self.game_over(left_acion_num, planner_inputs,local_map,infos)
       
       
			self.Return_actions += self.set_back_to_angle(60)
			self.steps += len(self.Return_actions )
			return self.Return_actions

#######################################################################
		else :
			if self.check_picked_obj :
				self.check_picked_obj = False 
				picking_obj_mask = self.seg.sem_seg_get_instance_mask_from_obj_type_largest_only(planner_inputs['list_of_actions'][self.pointer-1][0])
				if picking_obj_mask is not None :
					height = np.where(picking_obj_mask !=0)[0] 
					widths = np.where(picking_obj_mask !=0)[1]
					hei_center = np.mean(height)
					wd_center = np.mean(widths)
					if wd_center >= 100 and wd_center<=200 and hei_center >= 100 and hei_center<=230 :  # pick up desired obj
						self.picked_up_desired_obj = True
						self.Return_actions += self.set_back_to_angle(30)
						self.steps += len(self.Return_actions )
						return self.Return_actions
					else : 
						# cv2.imshow("tmp",picking_obj_mask[ :,:,np.newaxis ] *  self.img[:,:,::-1])
						# cv2.waitKey()
				
						self.converted_subgoal = True
						try :
							last_lowlevel_inthesame_high = 0
							prev_high_level =-1 
							for planner_input in planner_inputs['list_of_actions'] :
								if prev_high_level == planner_inputs['list_of_actions'][self.pointer-1][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer-1][3] :
									break
								last_lowlevel_inthesame_high +=1
								prev_high_level = planner_input[3]
		
							left_acion_num = last_lowlevel_inthesame_high -self.pointer 
		
							added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"Double_check")
							if added_action == "[['Parent', 'PutObject'], ['Target', 'PickupObject']]" : 
								recep = 'CounterTop'
								for recep_option in ['CounterTop','Desk','Sofa']:
									cn = self.total_cat2idx[recep_option]
									if local_map[cn+4,:,:].sum() > 0. :
										recep = recep_option
										break
								planner_inputs['list_of_actions'].insert(self.pointer, [planner_inputs['list_of_actions'][self.pointer-1][0], planner_inputs['list_of_actions'][self.pointer-1][1],planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
								planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 

								self.second_object.insert(self.pointer,self.second_object[self.pointer])
								self.second_object.insert(self.pointer,None)

								self.set_caution_point(planner_inputs,infos)
							else : 
								new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
								self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
						except : 
							left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
							self.game_over(left_acion_num, planner_inputs,local_map,infos)      

				else : 
					self.converted_subgoal = True
					try :
						last_lowlevel_inthesame_high = 0
						prev_high_level =-1 
						for planner_input in planner_inputs['list_of_actions'] :
							if prev_high_level == planner_inputs['list_of_actions'][self.pointer-1][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer-1][3] :
								break
							last_lowlevel_inthesame_high +=1
							prev_high_level = planner_input[3]
	
						left_acion_num = last_lowlevel_inthesame_high -self.pointer 
	
						added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"Double_check")
						if added_action == "[['Parent', 'PutObject'], ['Target', 'PickupObject']]": 
							recep = 'CounterTop'
							for recep_option in ['CounterTop','Desk','Sofa']:
								cn = self.total_cat2idx[recep_option]
								if local_map[cn+4,:,:].sum() > 0. :
									recep = recep_option
									break
							planner_inputs['list_of_actions'].insert(self.pointer, [planner_inputs['list_of_actions'][self.pointer-1][0], planner_inputs['list_of_actions'][self.pointer-1][1],planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 
							planner_inputs['list_of_actions'].insert(self.pointer, [recep, 'PutObject',planner_inputs['list_of_actions'][self.pointer-1][2],planner_inputs['list_of_actions'][self.pointer-1][3]]) 

							self.second_object.insert(self.pointer,self.second_object[self.pointer])
							self.second_object.insert(self.pointer,None)

							self.set_caution_point(planner_inputs,infos)
						else : 
							new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
							self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
					except : 
						left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
						self.game_over(left_acion_num, planner_inputs,local_map,infos)      

				self.Return_actions += self.set_back_to_angle(60)
				self.steps += len(self.Return_actions )
				return self.Return_actions
			return self.Return_actions




	####################### Before ######  Plan Act and preprocess (interact with the mapping module)    ##############################################
	def semexp_plan_act_and_preprocess_before_step(self, planner_inputs, goal_spotted,infos,local_map,simulator):
		#These are things from plan_act_and_preprocess
  
		
		self.pointer = planner_inputs['list_of_actions_pointer']
		self.current_subgoal = planner_inputs['list_of_actions'][self.pointer]
		self.lislist_of_actions = planner_inputs['list_of_actions']
  
		traversible, cur_start, cur_start_o = self.get_traversible(planner_inputs)

		self.cur_start = cur_start


		self.activate_lookDownUpLeft_count = False
		self.steps += 1
		self.moved_until_visible = False
		self.side_stepped = None
		self.sdroate_direction = None
		self.IN_opp_side_step = False

		self._visualize(planner_inputs)

		self.goal_success = False
		self.giveup_heat = False
		# keep_consecutive = False

		self.Return_actions = []

		#If statements and get Return_actions
		if self.prev_point != self.pointer: 
			print("current subgoal : ",str(self.pointer), planner_inputs['list_of_actions'][self.pointer])
			self.prev_point = self.pointer
		self.goal_success = False
		keep_consecutive = False
#######################################################################################################################################

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

			elif self.lookDownUpLeft_count ==2:
				#look down back to 45
				cur_hor = np.round(self.camera_horizon, 4)
				self.Return_actions += self.set_back_to_angle(60)
				action = "RotateLeft_90"
				self.Return_actions += self.va_interact_new("RotateLeft_90")

			
		
		# elif planner_inputs['consecutive_interaction'] != None:
		# 	if self.args.debug_local:
		# 		self.print_log("consec action")
			
		# 	target_object_type = planner_inputs['consecutive_target']
			
		# 	self.Return_actions += self.consecutive_interaction(planner_inputs['consecutive_interaction'], target_object_type)


###########################   0430 #########################################3  
		elif planner_inputs['consecutive_interaction'] != None:   ## before 
			 

			list_of_actions = planner_inputs['list_of_actions']
			pointer = planner_inputs['list_of_actions_pointer']

			if self.args.debug_local:
				self.print_log("consec action")
			
			target_object_type = planner_inputs['consecutive_target']
			
			self.Return_actions += self.consecutive_interaction(
				planner_inputs['consecutive_interaction'], target_object_type, list_of_actions ,pointer
			)	
##############################################################################3
			
		elif planner_inputs['consecutive_interaction'] == None and self.execute_interaction:
			#Do the deterministic interaction here
			list_of_actions = planner_inputs['list_of_actions']
			pointer = planner_inputs['list_of_actions_pointer']
			interaction = list_of_actions[pointer][1]
				
			if interaction == None:
				pass
# ############################################################################################################################			
# 			## 0430 remember sliced
# 			if 'Sliced' in self.goal_name and self.interaction_mask is None and self.sliced_mask_whenSliced is not None:
# 				self.interaction_mask = self.sliced_mask_whenSliced
# ##################################################################
			else:
				#################  Slice Knife Put  ################################s
				if interaction =="SliceObject" and list_of_actions[pointer][2]==list_of_actions[pointer+1][2] == "Cut" and\
					len(list_of_actions)-1 > pointer and list_of_actions[pointer+1][1] == "PutObject":
						bread_recep=self.seg.sem_seg_get_recep_cls_under_target(self.interaction_mask)
						if bread_recep is not None :
							
							planner_inputs['list_of_actions'].insert(self.pointer+1, [bread_recep, list_of_actions[pointer+1][1],planner_inputs['list_of_actions'][self.pointer+1][2],planner_inputs['list_of_actions'][self.pointer+1][3]]) 
							self.second_object.insert(self.pointer+1,None)
       
							del planner_inputs['list_of_actions'][self.pointer+2]
							del self.second_object[self.pointer+2]
							self.set_caution_point(planner_inputs,infos)
#################################################################################################################       
				self.Return_actions += self.va_interact_new(interaction, mask=self.interaction_mask)
	
			
		else:
			if self.prev_number_action !=0:
				number_action = self._plan(planner_inputs, self.steps, goal_spotted, planner_inputs['newly_goal_set'])
				action_dict = {
					0: "<<stop>>", 
					1: "MoveAhead_25", 
					2:"RotateLeft_90",
					3:"RotateRight_90", 
					4: "LookDown_90", 
					5:"LookDownUpLeft"}
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
				
				elif self.is_visible_from_mask(self.interaction_mask,stricer_visibility_dist=self.args.stricter_visibility): #Must be cautious pointers
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
							self.Return_actions +=self.va_interact_new("RotateLeft_90") #pass
							self.Return_actions +=self.va_interact_new("RotateRight_90")
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
							self.Return_actions +=self.va_interact_new("RotateLeft_90") #pass
							self.Return_actions +=self.va_interact_new("RotateRight_90")
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
						self.Return_actions += self.set_back_to_angle(60)
					else:
						self.Return_actions +=self.va_interact_new("RotateLeft_90") #pass
						self.Return_actions +=self.va_interact_new("RotateRight_90")

					self.activate_lookDownUpLeft_count = True

				else:
					self.Return_actions += self.va_interact_new(action)


########################### Attribute : Before PutSink  ########################
		if self.Return_actions[0][0]=="PutObject" and planner_inputs['list_of_actions'][self.pointer][0] == "SinkBasin":
			if (self.pointer ==0) or (planner_inputs['list_of_actions'][self.pointer -1][0] != 'SinkBasin' and planner_inputs['list_of_actions'][self.pointer -1][1] !='ToggleObjectOff'):
				mask =self.Return_actions[0][1]
				test_img =   mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
				test_img = self.crop_img_fitting_on_obj(test_img)

				###Sink는 빼는게 잘됨 
				test_img = self.enhance_image_brightness_and_sharpness(test_img,127)

				largest_lab = self.Check_similarity(test_img, "ToggleOn", obj_name="SinkBasin")
				if largest_lab == 0 :
					try :
						last_lowlevel_inthesame_high = 0
						prev_high_level =-1 
						for planner_input in planner_inputs['list_of_actions'] :
							if prev_high_level == planner_inputs['list_of_actions'][self.pointer][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer][3] :
								break
							last_lowlevel_inthesame_high +=1
							prev_high_level = planner_input[3]
	
						left_acion_num = last_lowlevel_inthesame_high -self.pointer 
      
						added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"attribute")
						if added_action == "['Faucet', 'ToggleObjectOff']" : 
							planner_inputs['list_of_actions'].insert(self.pointer, ['SinkBasin', 'ToggleObjectOff',planner_inputs['list_of_actions'][self.pointer][2],planner_inputs['list_of_actions'][self.pointer][3]]) 
							self.second_object.insert(self.pointer,None)
							self.set_caution_point(planner_inputs,infos)
							self.Return_actions = [('ToggleObjectOff',mask,None)]
						else : 
							new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
							self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
					except : 
						left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
						self.game_over(left_acion_num, planner_inputs,local_map,infos)

					
     
########################### Attribute : Before openMicrowave  ########################
		elif self.Return_actions[0][0]=="OpenObject" and planner_inputs['list_of_actions'][self.pointer][0] == "Microwave":
			if (self.pointer ==0) :
				mask =self.Return_actions[0][1]
				test_img =   mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
				test_img = self.crop_img_fitting_on_obj(test_img)
				test_img = self.enhance_image_brightness_and_sharpness(test_img,127)
				largest_lab = self.Check_similarity(test_img, "ToggleOn", obj_name="Microwave")
    
				if largest_lab == 0 :
					try : 
						last_lowlevel_inthesame_high = 0
						prev_high_level =-1 
						for planner_input in planner_inputs['list_of_actions'] :
							if prev_high_level == planner_inputs['list_of_actions'][self.pointer][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer][3] :
								break
							last_lowlevel_inthesame_high +=1
							prev_high_level = planner_input[3]
	
						left_acion_num = last_lowlevel_inthesame_high -self.pointer 
      
						added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"attribute")
						if added_action == "['Microwave', 'ToggleObjectOff']" :         
							planner_inputs['list_of_actions'].insert(self.pointer, ['Microwave', 'ToggleObjectOff',planner_inputs['list_of_actions'][self.pointer][2],planner_inputs['list_of_actions'][self.pointer][3]]) 
							self.second_object.insert(self.pointer,None)
							self.set_caution_point(planner_inputs,infos)
							self.Return_actions = [('ToggleObjectOff',mask,None)]
						else : 
							new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
							self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
					except : 
						left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
						self.game_over(left_acion_num, planner_inputs,local_map,infos)     
      
      
			elif (planner_inputs['list_of_actions'][self.pointer -1][0] != 'Microwave' and planner_inputs['list_of_actions'][self.pointer -1][1] !='ToggleObjectOff'): 
				mask =self.Return_actions[0][1]

				test_img =   mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
				test_img = self.crop_img_fitting_on_obj(test_img)
				test_img = self.enhance_image_brightness_and_sharpness(test_img,127)

				largest_lab = self.Check_similarity(test_img, self.task_type, obj_name="Microwave")
				if largest_lab == 0 :
					try : 
						last_lowlevel_inthesame_high = 0
						prev_high_level =-1 
						for planner_input in planner_inputs['list_of_actions'] :
							if prev_high_level == planner_inputs['list_of_actions'][self.pointer][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer][3] :
								break
							last_lowlevel_inthesame_high +=1
							prev_high_level = planner_input[3]
	
						left_acion_num = last_lowlevel_inthesame_high -self.pointer 
      
						added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"attribute")
						if added_action == "['Microwave', 'ToggleObjectOff']" :         
							planner_inputs['list_of_actions'].insert(self.pointer, ['Microwave', 'ToggleObjectOff',planner_inputs['list_of_actions'][self.pointer][2],planner_inputs['list_of_actions'][self.pointer][3]]) 
							self.second_object.insert(self.pointer,None)
							self.set_caution_point(planner_inputs,infos)
							self.Return_actions = [('ToggleObjectOff',mask,None)]
						else : 
							new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
							self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
					except : 
						left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
						self.game_over(left_acion_num, planner_inputs,local_map,infos)
     


########################### step count giv up ##################################

		elif planner_inputs['list_of_actions'][self.pointer][1] == "PickupObject" and planner_inputs['list_of_actions'][self.pointer][2] == "Clean" and self.steps_taken > 600 :
			if self.task_type in ['N Slices Of X In Y'] and planner_inputs['list_of_actions'][-1][2] == "Cut" :
				total_action_length = len(planner_inputs['list_of_actions'])
				for _ in range(self.pointer, total_action_length) :
					if planner_inputs['list_of_actions'][self.pointer][2] != "Cut":
						del planner_inputs['list_of_actions'][self.pointer]
						del self.second_object[self.pointer]
					else : break
				self.set_caution_point(planner_inputs,infos)
				self.goal_idx = self.total_cat2idx[planner_inputs['list_of_actions'][self.pointer][0]]
    
			elif self.task_type in ['N Cooked Slices Of X In Y'] :
				current_subgoal_highlevel = []
				hi_inx =[]
				for action_ in planner_inputs['list_of_actions'] :
					if action_[3] not in hi_inx:
						hi_inx.append(action_[3])
						current_subgoal_highlevel.append(action_[2])
				if current_subgoal_highlevel == ['Clean','Move','Cut','Heat'] :
					total_action_length = len(planner_inputs['list_of_actions'])
					for _ in range(self.pointer, total_action_length) :
						if planner_inputs['list_of_actions'][self.pointer][2] != "Cut":
							del planner_inputs['list_of_actions'][self.pointer]
							del self.second_object[self.pointer]
						else : break
					self.set_caution_point(planner_inputs,infos)
					self.goal_idx = self.total_cat2idx[planner_inputs['list_of_actions'][self.pointer][0]]

			elif self.task_type in ['Salad','Sandwich'] :
				current_subgoal_highlevel = []
				hi_inx =[]
				for action_ in planner_inputs['list_of_actions'] :
					if action_[3] not in hi_inx:
						hi_inx.append(action_[3])
						current_subgoal_highlevel.append(action_[2])
      
				if 	'Cut' in current_subgoal_highlevel:
					total_action_length = len(planner_inputs['list_of_actions'])
					for _ in range(self.pointer, total_action_length) :
						if planner_inputs['list_of_actions'][self.pointer][2] != "Cut":
							del planner_inputs['list_of_actions'][self.pointer]
							del self.second_object[self.pointer]
						else : break
					self.set_caution_point(planner_inputs,infos)
					self.goal_idx = self.total_cat2idx[planner_inputs['list_of_actions'][self.pointer][0]]
			

		return self.Return_actions







	####################### After ####### Plan Act and preprocess (interact with the mapping module)    ##############################################
	def semexp_plan_act_and_preprocess_after_step(self, planner_inputs, goal_spotted,infos,local_map,simulator) :
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
				
				# 0430
				visible = self.is_visible_from_mask(self.interaction_mask,stricer_visibility_dist=self.args.stricter_visibility)
    
				if not(pointer in self.caution_pointers):
					# self.execute_interaction = goal_spotted and visible 
					self.execute_interaction = visible 
				else: 
					if planner_inputs['list_of_actions'][pointer][0] not in [
					 	'Drawer',
					 	'SinkBasin',
					 	'Cabinet',
					 	'Fridge',
					 	'Microwave',
					 	'Safe',
						'Toaster'
       ############################################
						# 'CoffeeMachine',   
						# 'StoveKnob',
						# 'HousePlant'
      #############################################
					]:
						self.execute_interaction = visible
						# self.execute_interaction = goal_spotted and visible 
					else:
						whether_center = self.whether_center()
						# self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center 
						self.execute_interaction = visible and self.prev_number_action == 0 and whether_center 
						if self.IN_opp_side_step:
							# self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 
							self.execute_interaction = visible and self.prev_number_action == 0 
				self.PREV_side_step_order = 0
    
###############################################################################    
			elif self.PREV_side_step_order == 1 and not self.last_success:
				_, cur_start, cur_start_o = self.get_traversible(planner_inputs)
				# add collision
				if int(cur_start_o) == 0:
					self.collision_map[cur_start[0], cur_start[1] + 1:cur_start[1] + 6] = 1
				elif int(cur_start_o) == 90:
					self.collision_map[cur_start[0] + 1:cur_start[0] + 6, cur_start[1]] = 1
				elif int(cur_start_o) == -180:
					self.collision_map[cur_start[0], cur_start[1] - 5:cur_start[1]] = 1
				elif int(cur_start_o) == -90:
					self.collision_map[cur_start[0] - 5:cur_start[0], cur_start[1]] = 1
				else:
					print('No valid cur_start_o (not divisible by 90):', int(cur_start_o))
				self.print_log("side step moved and failed!")
###############################################################################			
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
				action = "RotateLeft_90"
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
				self.put_rgb_mask = None
				self.recep_mask = None
    
			elif self.last_action_ogn == "PutObject" and self.goal_success:
				self.picked_up = False
				self.picked_up_cat = None
				
				# self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.img)
				############ 0430
				self.put_rgb_mask = ((self.prev_rgb*self.interaction_mask[ :,:,np.newaxis ].astype(np.uint8)!=self.img*self.interaction_mask[ :,:,np.newaxis ].astype(np.uint8))*1.0)[:,:,0]
				self.recep_mask = self.interaction_mask
       			#########################################################
				self.picked_up_mask = None
    
	
			if self.last_action_ogn == "OpenObject" and self.last_success:
				self.open_mask = copy.deepcopy(self.interaction_mask)
			
			obs = obs_temp
			if self.last_action_ogn == "SliceObject" and self.goal_success:
				self.first_goal_sliced_state = True
				############ 0430
				# f1 = self.prev_rgb; f2 = self.img
				# diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
				# diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])Togg
				# diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
				# diff_mask = np.zeros((300,300))
				# diff_mask[diff1] =1.0 
				# diff_mask[diff2] =1.0 
				# diff_mask[diff3] =1.0 
				# diff_mask.astype(np.uint8)

				# self.sliced_mask_whenSliced = diff_mask
				# self.loc_whenSliced = cur_start
				# self.rgb_mask_whenSliced = self.img

			if 	self.last_action_ogn == "ToggleObjectOn" and self.goal_success  and self.goal_name == 'Faucet':	
				self.faucet_on_mask = self.interaction_mask

			if 	self.last_action_ogn == "ToggleObjectOff" and self.goal_success  and self.goal_name == 'Faucet':	
				self.faucet_on_mask = None
		
			
		elif planner_inputs['consecutive_interaction'] == None and self.execute_interaction:
			self.print_log("Fell into planner_inputs['consecutive_interaction'] == None and self.execute_interaction")
			#Do the deterministic interaction here
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
					self.put_rgb_mask = None
					self.recep_mask = None

				elif (self.last_action_ogn == "PutObject" and self.goal_success) :
					self.picked_up = False
					self.picked_up_cat = None

					# self.put_rgb_mask = self.seg.H.diff_two_frames(self.prev_rgb, self.img)
					# 0430
					self.put_rgb_mask = ((self.prev_rgb*self.interaction_mask[ :,:,np.newaxis ].astype(np.uint8)!=self.img*self.interaction_mask[ :,:,np.newaxis ].astype(np.uint8))*1.0)[:,:,0]
					self.recep_mask = self.interaction_mask
     				###################
					self.picked_up_mask = None
		
				if self.last_action_ogn == "OpenObject" and self.last_success:
					self.open_mask = copy.deepcopy(self.interaction_mask)
				
				obs = obs_temp

				#self.info = info
			if self.last_action_ogn == "SliceObject" and self.goal_success:
				self.first_goal_sliced_state = True
    
			if 	self.last_action_ogn == "ToggleObjectOff" and self.goal_success  and self.goal_name == 'Faucet':	
				self.faucet_on_mask = None
			
		else:
			if self.activate_lookDownUpLeft_count:
				self.lookDownUpLeft_count = 1
    
    
    
			else : 
				if self.last_action_ogn == "MoveAhead_25" and  not self.last_success:
						# add collision
					_, cur_start, cur_start_o = self.get_traversible(planner_inputs)
					if int(cur_start_o) == 0:
						self.collision_map[cur_start[0], cur_start[1] + 1:cur_start[1] + 6] = 1
					elif int(cur_start_o) == 90:
						self.collision_map[cur_start[0] + 1:cur_start[0] + 6, cur_start[1]] = 1
					elif int(cur_start_o) == -90:
						self.collision_map[cur_start[0] - 5:cur_start[0], cur_start[1]] = 1
					elif int(cur_start_o) == -180:
						self.collision_map[cur_start[0], cur_start[1] - 5:cur_start[1]] = 1
					else:
						print('No valid cur_start_o (not divisible by 90):', int(cur_start_o))					
					
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
			
			visible = self.is_visible_from_mask(self.interaction_mask,stricer_visibility_dist=self.args.stricter_visibility)
   
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


###################### 0430 ######################################################################################################
			# ## remember sliced
			# if 'Sliced' in self.goal_name and self.loc_whenSliced is not None and cur_start == self.loc_whenSliced:
			# 	rgb_mask_diff = self.seg.H.diff_two_frames(self.rgb_mask_whenSliced, self.event.frame)
			# 	npsum = np.sum(rgb_mask_diff)
			# 	if npsum < 6000:
			# 		visible = True
############################################################################################################################
###############################################################################################################################
			###################  Receptacles Observation  : DTA
			if pointer < len(list_of_actions) :
				if planner_inputs['list_of_actions'][pointer][1] == "OpenObject" and planner_inputs['list_of_actions'][pointer+1][1] in ["PickupObject","SliceObject"] \
					and planner_inputs['list_of_actions'][pointer][3] == planner_inputs['list_of_actions'][pointer+1][3] :
         
					small_target_mask=self.seg.sem_seg_get_instance_mask_from_obj_type(list_of_actions[pointer+1][0])
					visible = self.is_visible_from_mask(small_target_mask, stricer_visibility_dist=self.args.stricter_visibility)

					#현재 물건이 보이거나, 이미 본적이 있어서 map에 저장된 경우 
					if visible or np.sum(planner_inputs['class_map'][self.total_cat2idx[list_of_actions[pointer+1][0]]])>0: 


						try :
							last_lowlevel_inthesame_high = 0
							prev_high_level =-1 
							for planner_input in planner_inputs['list_of_actions'] :
								if prev_high_level == planner_inputs['list_of_actions'][self.pointer][3] and planner_input[3] !=planner_inputs['list_of_actions'][self.pointer][3] :
									break
								last_lowlevel_inthesame_high +=1
								prev_high_level = planner_input[3]
		
							left_acion_num = last_lowlevel_inthesame_high -self.pointer 
		
							added_action, revised_result = self.update_and_refine_actions_by_GPT_low(left_acion_num,planner_inputs['list_of_actions'],"Observation")
							if added_action == "[['Recep', 'OpenObject'], ['Recep', 'CloseObject']]" : ##### 
								self.execute_interaction = visible
								self.goal_idx = self.total_cat2idx[list_of_actions[pointer][0]]

								planner_inputs['list_of_actions'][pointer]=planner_inputs['list_of_actions'][pointer+1]
								del planner_inputs['list_of_actions'][pointer+1]
								del planner_inputs['list_of_actions'][pointer+1] 
			
								del self.second_object[pointer+1] 
								del self.second_object[pointer+1] 

								self.set_caution_point(planner_inputs,infos)

								self.interaction_mask =small_target_mask
								self.print_log("Converted : pointer increased goal name ",planner_inputs['list_of_actions'][pointer])
							else : 
								new_lowactions = convert(self.edh_instance['instance_id'], ast.literal_eval(revised_result), picked_up_obj=planner_inputs['list_of_actions'][self.pointer-1][0])
								self.revise_same_highlevel_following_actions(left_acion_num, planner_inputs,new_lowactions,local_map,infos)						
						except : 
							left_acion_num = len(planner_inputs['list_of_actions']) - self.pointer -1
							self.game_over(left_acion_num, planner_inputs,local_map,infos)

					else :
						visible = self.is_visible_from_mask(self.interaction_mask, stricer_visibility_dist=self.args.stricter_visibility)
####################################################################################################################################################

			
			self.execute_interaction = False
			if not(pointer in self.caution_pointers):   # Pickup 일때 항상 !!
				self.execute_interaction =  visible 
				########### Spa relation
				if visible and planner_inputs['list_of_actions'][pointer][1] =="PickupObject" and self.second_object[pointer] is not None:
					if self.second_object[pointer]  == self.seg.sem_seg_get_recep_cls_under_target(self.interaction_mask) :
						print("Already put on desired Place!!!!")
						self.execute_interaction = False
						# self.pass_obj = True
      
						cn = self.total_cat2idx[planner_inputs['list_of_actions'][pointer][0] ]
						basic_map = np.ones_like(planner_inputs["class_map"], dtype=int)  * 10
						basic_map[cn,:,:]=  0 
						planner_inputs["class_map"]=  basic_map
      
						# cn = self.total_cat2idx[planner_inputs['list_of_actions'][pointer][0] ]
						# planner_inputs["class_map"][cn,:,:]=  0 

				#################################################3
    
    
			else: #caution pointers
				if list_of_actions[pointer][0] not in [
				 	'Drawer',
				 	'SinkBasin',
				 	'Sink',
				 	'Cabinet',
				 	'Fridge',
				 	'Microwave',
				 	'Safe',
				 	'Dresser'
					'Toaster'
       ############################################
					# 'HousePlant'
					# 'CoffeeMachine',   
					# 'StoveKnob'
      #############################################
				]:
					self.execute_interaction = visible
					# self.execute_interaction = goal_spotted and visible
				else:
					whether_center = self.whether_center()
					#self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 and whether_center
					self.execute_interaction = visible and self.prev_number_action == 0 and whether_center
			
					if self.IN_opp_side_step:
						# self.execute_interaction = goal_spotted and visible and self.prev_number_action == 0 
						self.execute_interaction = visible and self.prev_number_action == 0 

		delete_lamp = (self.mvb_num_in_cycle !=0)  and \
      					(self.goal_name == 'FloorLamp') and \
          				(self.action_received == "ToggleObjectOn")
		if self.args.no_delete_lamp:
			delete_lamp = False

		if not(self.moved_until_visible):
			self.mvb_num_in_cycle = 0
			self.mvb_which_cycle = 0



		self.rotate_aftersidestep =self.sdroate_direction
		next_step_dict = {'keep_consecutive': self.giveup_heat, 'view_angle': self.camera_horizon, \
								  'picked_up': self.picked_up, 'errs': self.errs, 'steps_taken': self.steps_taken, 'broken_grid':self.broken_grid, 'actseq':{(self.args.from_idx + self.scene_pointer* self.args.num_processes + self.rank, self.traj_data['instance_id']): self.actions[:1000]},
								  'logs':self.logs,  'current_goal_sliced':self.cur_goal_sliced, 'move_until_visible_cycled': self.mvb_which_cycle != 0 and self.mvb_num_in_cycle==0, 'delete_lamp': delete_lamp,
								  'fails_cur': self.fails_cur, 'edh_instance_true': self.edh_instance['instance_id'], 'total_num_pointers': len(planner_inputs['list_of_actions']), 'list_of_actions': planner_inputs['list_of_actions'], 
          						  'class_map'  : planner_inputs["class_map"]}
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
		return self.obs, self.info, self.goal_success, next_step_dict


	###############################################################################################
	#All the actions
	def va_interact_new(self, action, last_success_special_treatment = None, mask=None):		
		if action in ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "PourObject", "SliceObject"] :
			print("Returning ", [(action, mask, last_success_special_treatment)])  
			if mask is None or np.sum(mask)<1:
				return [('LookUp_0', None, None)]   #JY
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
	
			else:
				times_30 = int((angle_arg-view_angle_copy)/30)
				for i in range(times_30):
					Return_actions +=self.va_interact_new("LookDown_30", last_success_special_treatment =last_success_special_treatment)

			return Return_actions

		else:
			Return_actions += self.va_interact_new("RotateLeft_90" , last_success_special_treatment =last_success_special_treatment)
			Return_actions += self.va_interact_new("RotateRight_90" , last_success_special_treatment =last_success_special_treatment)
			return Return_actions

	def move_until_visible(self):
		Return_actions = []
		if not(self.rotate_aftersidestep is None) and abs(int(self.camera_horizon)  - 0) <5 : #rotated after sidestep but no longer visible
			return self.set_back_to_angle(60)
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
					#if Return_actions == []:
					#	Return_actions += self.va_interact_new("LookUp_0", last_success_special_treatment = "NoUpdate")
					self.print_log("RETURN ACTIONS OF move_until_visible are ", Return_actions)
					self.print_log("Move until visible horizon is ", self.camera_horizon)
					#self.print_log("REAL HORIZON IS ", self.env.last_event.metadata['agent']['cameraHorizon'])

					#self.consecutive_steps = False
					#self.final_sidestep = False
				self.move_until_visible_order +=1
				
			else:
				self.print_log("Move until visible horizon is ", self.camera_horizon)

				Return_actions += self.set_back_to_angle(60)

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
				Return_actions +=self.va_interact_new("RotateRight_90", last_success_special_treatment = "FalseUpdate")
				Return_actions +=self.va_interact_new("RotateLeft_90", last_success_special_treatment = "FalseUpdate")

			else:
				Return_actions +=self.va_interact_new("MoveAhead_25")

			self.side_step_order =2

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


	# def consecutive_interaction(self, interaction, target_object_type):
	# 	Return_actions = []
	# 	if interaction == "PutObject" and self.last_action_ogn == "OpenObject":
	# 		self.interaction_mask =self.open_mask
	# 	elif interaction == "CloseObject":
	# 		self.interaction_mask =self.open_mask
	# 	elif self.args.use_sem_seg:
	# 		self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(target_object_type)
	# 	else:
	# 		self.interaction_mask = self.seg.get_instance_mask_from_obj_type(target_object_type)

	# 	Return_actions += self.va_interact_new(interaction, mask =self.interaction_mask)

	# 	return Return_actions
 
#  ################################# 0430 ###############################################
# 	def consecutive_interaction(self, interaction, target_object_type,list_of_actions ,pointer):
# 		Return_actions = []
  

# 		if self.args.use_sem_seg:
# 			self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(target_object_type)
# 		else:
# 			self.interaction_mask = self.seg.get_instance_mask_from_obj_type(target_object_type)


# 		if np.sum(self.interaction_mask) ==None :
# 			if interaction == "PutObject" and self.last_action_ogn == "OpenObject":
# 				self.interaction_mask =self.open_mask
# 			elif interaction == "CloseObject":
# 				self.interaction_mask =self.open_mask

# 			elif self.last_action_ogn == "PickupObject" and not(self.last_success):
# 				if pointer > 0:
# 					## mrecep mask
# 					if list_of_actions[pointer-1][1] =='PutObject' and list_of_actions[pointer][0] == list_of_actions[pointer-1][0]  :
# 						self.interaction_mask = self.recep_mask
# 						overlap = self.interaction_mask ==  self.put_rgb_mask
# 						self.interaction_mask[overlap] = 0
# 					else :
# 						self.interaction_mask = self.put_rgb_mask
						
# 				else :
# 					##  use mask when obj is put in clean, cool, heat task 
# 					self.interaction_mask = self.put_rgb_mask



# 		Return_actions += self.va_interact_new(interaction, mask =self.interaction_mask)

# 		return Return_actions


 ################################# 0430 ###############################################
	def consecutive_interaction(self, interaction, target_object_type,list_of_actions ,pointer):
		Return_actions = []
		print(np.sum(self.interaction_mask))
		prev_mask = self.interaction_mask 
		if self.args.use_sem_seg:
			self.interaction_mask = self.seg.sem_seg_get_instance_mask_from_obj_type(target_object_type)
		else:
			self.interaction_mask = self.seg.get_instance_mask_from_obj_type(target_object_type)



		if interaction == "PutObject" and self.last_action_ogn == "OpenObject" and np.sum(self.open_mask)!=None:
			self.interaction_mask =self.open_mask
		if interaction == "PutObject" and self.last_action_ogn == "ToggleObjectOff" and np.sum(prev_mask)!=None and target_object_type == 'SinkBasin' and self.interaction_mask is None:  ########
			print("-----------Try put and toggleoff -----------------")
			# self.interaction_mask =self.recep_mask
			self.interaction_mask = prev_mask

			test_img =   self.interaction_mask[ :,:,np.newaxis ] *  self.img[:,:,::-1]
			# cv2.imshow("Sinkput_mask",test_img.astype(dtype='uint8'))
			# cv2.waitKey(3000)
			# cv2.destroyWindow("Sinkput_mask") 
   
		elif interaction == "CloseObject"and np.sum(self.open_mask)!=None:
			self.interaction_mask =self.open_mask

		# elif self.last_action_ogn == "PickupObject" and not(self.last_success)and np.sum(self.recep_mask)!=None and np.sum(self.put_rgb_mask)!=None:
		# 	if pointer > 0:
		# 		## mrecep mask
		# 		if list_of_actions[pointer-1][1] =='PutObject' and list_of_actions[pointer][0] == list_of_actions[pointer-1][0]  :
		# 			self.interaction_mask = self.recep_mask
		# 			overlap = self.interaction_mask ==  self.put_rgb_mask
		# 			self.interaction_mask[overlap] = 0
		# 		else :
		# 			self.interaction_mask = self.put_rgb_mask
					
		# 	else :
		# 		##  use mask when obj is put in clean, cool, heat task 
		# 		self.interaction_mask = self.put_rgb_mask
    
    # ##########################################################################################################################################################
	# 	elif interaction == "PickupObject" and np.sum(self.put_rgb_mask)!=None and np.sum(self.recep_mask)!=None and np.sum(self.interaction_mask) ==None:
	# 		if pointer > 0:
	# 			## mrecep mask
	# 			if list_of_actions[pointer-1][1] =='PutObject' and list_of_actions[pointer][0] == list_of_actions[pointer-1][0]  :
	# 				self.interaction_mask = self.recep_mask
	# 				overlap = self.interaction_mask ==  self.put_rgb_mask
	# 				self.interaction_mask[overlap] = 0
	# 			else :
	# 				self.interaction_mask = self.put_rgb_mask
					
	# 		else :
	# 			##  use mask when obj is put in clean, cool, heat task 
	# 			self.interaction_mask = self.put_rgb_mask


	# 	elif interaction == "PickupObject" and np.sum(self.put_rgb_mask)!=None and np.sum(self.interaction_mask) == None:
	# 		self.interaction_mask = self.put_rgb_mask
	# ###########################################################################################################################################
    ##########################################################################################################################################################
		elif interaction == "PickupObject" and np.sum(self.put_rgb_mask)!=None and np.sum(self.recep_mask)!=None :
			if pointer > 0:
				## mrecep mask
				if list_of_actions[pointer-1][1] =='PutObject' and list_of_actions[pointer][0] == list_of_actions[pointer-1][0]  :
					self.interaction_mask = self.recep_mask
					overlap = self.interaction_mask ==  self.put_rgb_mask
					self.interaction_mask[overlap] = 0
				else :
					self.interaction_mask = self.put_rgb_mask
					
			else :
				##  use mask when obj is put in clean, cool, heat task 
				self.interaction_mask = self.put_rgb_mask


		elif interaction == "PickupObject" and np.sum(self.put_rgb_mask)!=None:
			self.interaction_mask = self.put_rgb_mask
	###########################################################################################################################################


		elif interaction== "ToggleObjectOff" and  np.sum(self.faucet_on_mask)!=None  and target_object_type == 'Faucet':	
			self.interaction_mask = self.faucet_on_mask

		Return_actions += self.va_interact_new(interaction, mask =self.interaction_mask)

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
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2) 

		# Get curr loc
		self.curr_loc = [start_x, start_y, start_o]
		
		
		r, c = start_y, start_x
		start = [int(r * 100.0/self.args.map_resolution - gx1), int(c * 100.0/self.args.map_resolution - gy1)]
		map_pred = np.rint(planner_inputs['map_pred'])
		start = pu.threshold_poses(start, map_pred.shape)

		self.visited[gx1:gx2, gy1:gy2][start[0]-0:start[0]+1, start[1]-0:start[1]+1] = 1
		
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
		# elif wd >= 65 and wd<=235:
		elif wd >= 100 and wd<=200:
			return True
		else:
			return False

	def get_traversible(self, planner_inputs):
		args = self.args
		map_pred = np.rint(planner_inputs['map_pred'])
		grid = map_pred
		

		# Get pose prediction and global policy planning window
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
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

		traversible = skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem) != True
		traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		traversible[int(start[0]-x1)-1:int(start[0]-x1)+2, int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
		if not(traversible[start[0], start[1]]):
			print("Not traversible, step is  ", self.steps_taken)

		traversible = add_boundary(traversible)

		# obstacle dilation
		traversible = 1 - traversible
		selem = skimage.morphology.disk(1)
		traversible = skimage.morphology.binary_dilation(traversible, selem) != True
		traversible = traversible * 1.
		
		
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
		start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
		gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
		planning_window = [gx1, gx2, gy1, gy2]

		# Get curr loc
		r, c = self.last_loc[1], self.last_loc[0]
		prev = [int(r * 100.0/args.map_resolution - gx1), int(c * 100.0/args.map_resolution - gy1)]
		prev = pu.threshold_poses(prev, map_pred.shape)
  
		self.curr_loc = [start_x, start_y, start_o]
		r, c = start_y, start_x
		start = [int(r * 100.0/args.map_resolution - gx1), int(c * 100.0/args.map_resolution - gy1)]
		start = pu.threshold_poses(start, map_pred.shape)

		self.visited[gx1:gx2, gy1:gy2][min(prev[0],start[0]):max(prev[0],start[0])+1,start[1]] = 1
		self.visited[gx1:gx2, gy1:gy2][start[0],min(prev[1],start[1]):max(prev[1],start[1])+1] = 1
		
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

		self.goal_visualize = goal
		stg, stop, whether_real_goal = self._get_stg(
			map_pred,
			start,
			np.copy(goal),
			planning_window,
			planner_inputs['found_goal'],
			xy.tolist(),
			step,
			planner_inputs['exp_pred'],
			goal_spotted,
			newly_goal_set,
			planner_inputs['list_of_actions_pointer'],
		)
  
		# Deterministic Local Policy
		if stop and whether_real_goal:
			action = 0


#####################################################################################################################
		# ## 0224 remember sliced
		# elif 'Sliced' in self.goal_name and start == self.loc_whenSliced:
		# 	action = 0
###########################################################################################################################


		elif stop:
			if self.action_5_count < 4:
				action = 5 #lookdown, lookup, left
				self.action_5_count +=1
			else:
				action = 2
		else:
			(stg_x, stg_y) = stg
			angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
			angle_agent = (start_o)%360.0
			if angle_agent > 180:
				angle_agent -= 360

			relative_angle = (angle_agent - angle_st_goal)%360.0
			if relative_angle > 180:
				relative_angle -= 360


			isFrees = {
				90: (self.collision_map[start[0] + 1:start[0] + 6, start[1]] > 0.5).sum() == 0,
				0: (self.collision_map[start[0], start[1] + 1:start[1] + 6] > 0.5).sum() == 0,
				-90: (self.collision_map[start[0] - 5:start[0], start[1]] > 0.5).sum() == 0,
				-180: (self.collision_map[start[0], start[1] - 5:start[1]] > 0.5).sum() == 0,
			}
			isFrontFree = isFrees[int(start_o)]


			if isFrontFree:
				if relative_angle > 45:
					action = 3  # Right
				elif relative_angle < -45:
					action = 2  # Left
				else:
					action = 1
			else:
				if relative_angle >= 0:
					action = 3
				else:
					action = 2

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

		##########################     discrete   #################################3 
		traversible = skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem) != True
		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		traversible[int(start[0]-x1), int(start[1]-y1)] = 1
		#################################################################
		traversible = add_boundary(traversible)

		##########################     discrete     #################################
		# obstacle dilation
		# traversible = (skimage.morphology.binary_dilation(1 - traversible, skimage.morphology.square(5)) != True) * 1.
		traversible = (skimage.morphology.binary_dilation(1 - traversible, skimage.morphology.square(1)) != True) * 1.
		# traversible = (skimage.morphology.binary_dilation(1 - traversible, skimage.morphology.square(2)) != True) * 1.
  
  
		# mask only explored area to traversible
		_traversible = traversible.copy()
		_traversible = (_traversible > 0.7).astype(_traversible.dtype)

		_traversible[1:-1, 1:-1][self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
		_traversible[1:-1, 1:-1][self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		_traversible[1:-1, 1:-1][int(start[0] - x1), int(start[1] - y1)] = 1

		traversible_clusters = skimage.morphology.label(_traversible, connectivity=2)
		traversible = (traversible_clusters == traversible_clusters[start[0]+1, start[1]+1]) * 1.

		grid_pattern = np.zeros_like(traversible)
  
		for i in range(grid_pattern.shape[0]):
			if i % 5 == 0:
				grid_pattern[i+1,:] = 1
		for j in range(grid_pattern.shape[1]):
			if j % 5 == 0:
				grid_pattern[:,j+1] = 1
		traversible *= grid_pattern
		traversible[0,:] = 0
		traversible[-1,:] = 0
		traversible[:,0] = 0
		traversible[:,-1] = 0
		for i in range(traversible.shape[0]):
			if i % 5 == 0:
				for j in range(0, traversible.shape[1], 5):
					if j % 5 == 0:
						if traversible[i+1,j+1] > 0.5 and traversible[i+1,j+6] > 0.5:
							if (traversible[i+1,j+2:j+5+1] > 0.5).sum() < 4:
								traversible[i+1,j+2:j+5+1] = 0
						else:
							traversible[i+1,j+2:j+5+1] = 0
		for j in range(traversible.shape[1]):
			if j % 5 == 0:
				for i in range(0, traversible.shape[0], 5):
					if i % 5 == 0:
						if traversible[i+1,j+1] > 0.5 and traversible[i+6,j+1] > 0.5:
							if (traversible[i+2:i+5+1,j+1] > 0.5).sum() < 4:
								traversible[i+2:i+5+1,j+1] = 0
						else:
							traversible[i+2:i+5+1,j+1] = 0


		traversible[1:-1, 1:-1][self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
		traversible[1:-1,1:-1][self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
		traversible[1:-1,1:-1][int(start[0]-x1), int(start[1]-y1)] = 1

		traversible_clusters = skimage.morphology.label(traversible, connectivity=1)
		traversible = (traversible_clusters == traversible_clusters[start[0] + 1, start[1] + 1]) * 1.

		import random
		if goal.shape == (240, 240):
			goal = add_boundary(goal, value=0)
   
    		#################################################################
		# move goal to the closest one
		def get_nearest_goal(goal, traversible):
			if goal.shape == (240, 240):
				goal = add_boundary(goal, value=0)
			_goal = copy.deepcopy(goal)	

			
			for _ in range(9):
				__goal = skimage.morphology.binary_erosion(_goal, skimage.morphology.disk(1))
				if __goal.sum() > 0:
					_goal = __goal

			goal = _goal
			new_goal = np.zeros_like(goal)
			for goal_coord in np.stack(np.nonzero(goal), axis=-1):
				coords = np.stack(np.nonzero(traversible > 0.5), axis=-1)
				coords = coords[(coords % 5 == 1).all(axis=-1)]
				idx = np.argmin((abs(coords - goal_coord)).sum(axis=1))
				new_goal[coords[idx][0], coords[idx][1]] = 1
			return new_goal
		goal = get_nearest_goal(goal, traversible)
		##################################################################
 
		#####################################################################
		
	############################################################################################################################  
		## go back
		if self.current_subgoal[0] in self.interacted_obj_list and self.prev_confirm_loc_obj != self.current_subgoal[0]:
			print("------------Go back!!!!!!!!!!!!!!!!!!!!!!!!!-------------------1")  ##################
			self.prev_confirm_loc_obj = self.current_subgoal[0]
			goal[self.interacted_obj_list[self.current_subgoal[0]]['waypoint'][0],self.interacted_obj_list[self.current_subgoal[0]]['waypoint'][1]] = 1  
############################################################################################################################
		original_goal = copy.deepcopy(goal)
	   
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
		 
		
		
		planner = FMMPlanner(self, traversible, self.args, step_size=self.args.step_size, start=start)
		
		if not(self.prev_wall_goal is None) and not(goal_found):
			if self.args.debug_local:
				self.print_log("wall goal")
			goal = self.prev_wall_goal
			self.goal_visualize = delete_boundary(goal)
			
		if self.dilation_deg!=0: 
			if self.args.debug_local:
				self.print_log("dilation added")
			goal = CH._add_cross_dilation(goal, self.dilation_deg, 0)#3)
			
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
					traversible_points = np.stack(np.nonzero(traversible), axis=-1)
					goal_point = np.stack(np.nonzero(goal), axis=-1)
					distances = (abs(traversible_points - goal_point)).sum(axis=-1)
					idx = np.argmin(distances)
					w_goal, h_goal = traversible_points[idx]

					new_goal = np.zeros_like(goal)
					new_goal[w_goal,h_goal] = 1
					new_goal = get_nearest_goal(new_goal, traversible)
					goal = new_goal
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

		state = [start[0] - x1 + 1, start[1] - y1 + 1]
		decrease_stop_cond =0
		if self.dilation_deg >= 6:
			decrease_stop_cond = 0.2 #0.2 #decrease to 0.2 (7 grids until closest goal)
		stg_x, stg_y, _, stop = planner.get_short_term_goal(
      		state, 
        	found_goal = found_goal, 
         	decrease_stop_cond=decrease_stop_cond)
  
		self.fmm_dist = planner.fmm_dist
		
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
			self.Return_actions = self.set_back_to_angle(60)
		
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
   

######################################################
			height = np.where(mask !=0)[0] 
			hei_center = np.mean(height)
			if self.lislist_of_actions[self.pointer][0] == "Toaster" or self.lislist_of_actions[self.pointer][1] == "SliceObject" :
				return (hei_center > 75)and (min_depth <= stricer_visibility_dist)

			else :
				return (hei_center > 30)and (min_depth <= stricer_visibility_dist)
###################################################################################
			# return (min_depth <= stricer_visibility_dist)

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

		# cv2.imshow("Sem Map", self.semantic[:,:,[2,1,0,3]])
		# cv2.waitKey(1)

		if self.args.visualize:
			semantic_vis = cv2.resize(self.semantic, (500, 500)) 
			cv2.imshow("Sem Map", semantic_vis[:,:,[2,1,0,3]])
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
