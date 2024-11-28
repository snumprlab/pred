#import sys
#sys.path.insert(0,'.')

import json
from alfred_training_pic.env.thor_env import ThorEnv
from sanity_check_model import SampleModel
import argparse
#from main_model import MainModel 
from FILM_model import FILMModel
#from controller_without_sim import SemExp_ControllerWithoutSim
import pickle
import cv2
import faulthandler

faulthandler.enable()

def noop():
    '''
    do nothing
    '''
    super().step(dict(action='Pass'))
    

def setup_scene(env, traj_data, r_idx, reward_type='dense'):
	'''
	intialize the scene and agent from the task info
	'''
	# scene setup
	scene_num = traj_data['scene']['scene_num']
	object_poses = traj_data['scene']['object_poses']
	dirty_and_empty = traj_data['scene']['dirty_and_empty']
	object_toggles = traj_data['scene']['object_toggles']

	scene_name = 'FloorPlan%d' % scene_num
	env.reset(scene_name)
	env.restore_scene(object_poses, object_toggles, dirty_and_empty)

	# initialize to start position
	env.step(dict(traj_data['scene']['init_action']))

	# print goal instr
	print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

def load_traj(scene_name):
	json_dir = 'alfred_data_all/json_2.1.0/' + scene_name['task'] + '/pp/ann_' + str(scene_name['repeat_idx']) + '.json'
	traj_data = json.load(open(json_dir))
	return traj_data

def main():
	#main_model = MainModel()
	#main_model = SampleModel(1,1, [])
	env = ThorEnv()
	main_model = FILMModel(1,1, env, [])
	files = json.load(open("alfred_data_small/splits/oct21.json"))['tests_unseen'][:100]#[args.eval_split][args.from_idx:args.to_idx]
	
	num_steps = 300
	for f in files:
		traj_data = load_traj(f); r_idx = traj_data['repeat_idx']
		setup_scene(env, traj_data, r_idx) 
		print("Starting real horizon is ", env.last_event.metadata['agent']['cameraHorizon'])
		#main_model.start_new_edh_instance(traj_data, cv2.cvtColor(env.last_event.frame, cv2.COLOR_RGB2BGR))
		main_model.start_new_edh_instance(traj_data, env.last_event.frame)
		action = None
		while not(action == 'Stop') :
			#action, mask = main_model.get_next_action(cv2.cvtColor(env.last_event.frame, cv2.COLOR_RGB2BGR), traj_data, action)
			action, mask = main_model.get_next_action(env.last_event.frame, traj_data, action)
			#main_model.num_sim_steps +=1
			#print("action is ", action); print("mask is ", mask)
			success, _, _, err, _ = env.va_interact(action, interact_mask=mask)
			#if not (mask is None):
			#	pickle.dump(mask, open("temp_pickles/mask_" + str(t) +".p", 'wb'))
			print("Action taken is ", action)
			print("Action success is ", success)
		print("Went out of while")




if __name__ == "__main__":
	#parser = argparse.ArgumentParser()
	#parser.add_argument("--seed", type=int, default=1, help="Random seed")
	#args = parser.parse_args()
	#main(args)
	main()
	print("All finsihed!")
