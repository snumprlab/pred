#from main_class_sanity_sanity import ClassMainSub
from main_class_sanity_init import ClassMain
from arguments import get_args
import torch
from envs import make_vec_envs
from datetime import datetime
import json
from agents.sem_exp_thor import Sem_Exp_Env_Agent_Thor



#main_args =  get_args()
# m = ClassMain()
# m.after_step_taken_initial()
# for i in range(3):
# 	m.after_step()

#c = ClassMainSub()
if __name__ == '__main__':
	global args
	global envs
	args = get_args()
	args.dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	args.device = torch.device("cuda:" + args.which_gpu if args.cuda else "cpu")
	args.skip_indices = {}
	torch.set_num_threads(1)
	#envs = make_vec_envs(args)
	rank = 0
	files = json.load(open("alfred_data_small/splits/oct21.json"))[args.eval_split][args.from_idx:args.to_idx]
	scene_names = files
	envs = Sem_Exp_Env_Agent_Thor(args, scene_names, rank) 

	c = ClassMain(args, envs)
	#print("Class initialized!")
	c.after_step_taken_initial()
	returned = False
	while returned == False:
		returned = c.after_step()