#Class that outputs prediction given models 
import os 
teach_dir = os.environ['TEACH_DIR']
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import sys
sys.path.append(os.path.join(teach_dir, os.environ["FILM_model_dir"], 'models/instructions_processed_LP'))
#from . import Bert_Classes
from Bert_Classes import TaskPredictionType, TaskPredictionObjTarget, TaskPredictionParentTarget
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import torch


class GetArguments:
	def __init__(self):
		Types_we_want = sorted(['Clean All X', 'N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Put All X In One Y', 'Put All X On Y'])
		self.Types_we_want_obj = {t: i for i, t in enumerate(Types_we_want)}

		Types_we_want_parent = sorted(['N Cooked Slices Of X In Y', 'N Slices Of X In Y', 'Put All X In One Y', 'Put All X On Y'])
		self.Types_we_want_parent = {t: i for i, t in enumerate(Types_we_want_parent)}

		self.tokenizer = AutoTokenizer.from_pretrained("/tmp/bert-base-uncased_I_downloaded")#("bert-base-uncased")#("/tmp/bert-base-uncased_I_downloaded")#

		pickle_dir = os.path.join(teach_dir,'FILM_model/models/instructions_processed_LP', 'BERT_models/bert_pickles')
		task_type_to_idx = pickle.load(open(os.path.join(pickle_dir , 'task_type_to_idx.p'), 'rb'))
		obj_count_to_idx = pickle.load(open(os.path.join(pickle_dir , 'obj_count_to_idx.p'), 'rb'))
		obj_target_to_idx = pickle.load(open(os.path.join(pickle_dir, 'obj_target_to_idx.p'), 'rb'))
		parent_target_to_idx = pickle.load(open(os.path.join(pickle_dir, 'parent_target_to_idx.p'), 'rb'))

		self.task_idx_2_type = {v:k for k,v in task_type_to_idx.items()}
		self.obj_idx_2_obj = {v:k for k,v in obj_target_to_idx.items()}
		self.parent_idx_2_parent = {v:k for k,v in parent_target_to_idx.items()}

		best_type_model = os.path.join(teach_dir, os.environ["FILM_model_dir"],'models/instructions_processed_LP', 'BERT_models/Type_epoch_179_acc_0.89.pt')
		best_obj_target_model = os.path.join(teach_dir, os.environ["FILM_model_dir"],'models/instructions_processed_LP', 'BERT_models/Obj_epoch_128_acc_0.97.pt')
		best_parent_model = os.path.join(teach_dir, os.environ["FILM_model_dir"],'models/instructions_processed_LP', 'BERT_models/Parent_epoch_172_acc_0.85.pt')
		#Load models
		self.type_model = TaskPredictionType()
		self.type_model.load_state_dict(torch.load(best_type_model, map_location=torch.device('cuda')))
		self.type_model = self.type_model.cuda()
		self.type_model.eval()


		self.obj_model = TaskPredictionObjTarget()
		self.obj_model.load_state_dict(torch.load(best_obj_target_model, map_location=torch.device('cuda')))
		self.obj_model = self.obj_model.cuda()
		self.obj_model.eval()

		self.parent_model = TaskPredictionParentTarget()
		self.parent_model.load_state_dict(torch.load(best_parent_model, map_location=torch.device('cuda'))) #success
		#Let's just do this for now and retrain
		self.parent_model = self.parent_model.cuda()
		self.parent_model.eval()

	def only_get_commnader_dialog_history(self, edh_instance):
		return_list = []
		for e in edh_instance["dialog_history"]:
			if e[0] == 'Commander':
				lowered = e[1].lower()
				if lowered[-1] != '.':
					lowered += '.'
				return_list.append(lowered)
		return ' '.join(return_list)


	def add_task_type_prompting(self, text, task_type):
		return_text = 'Task is ' + task_type + '. ' + text
		return return_text 

	# def get_type_pred(self, commander_dialog, type_model):

	# def get_obj_pred(self, commander_dialog, predicted_type, obj_model):

	# def get_parent_pred(self, commander_dialog, predicted_type, parent_model):

	def get_pred(self, edh_instance):
		#First get type
		text = self.only_get_commnader_dialog_history(edh_instance)
		tok = self.tokenizer(text, truncation=True, padding='max_length')

		type_pred = self.type_model(torch.tensor([tok['input_ids']]).cuda())
		type_pred = self.task_idx_2_type[torch.argmax(type_pred, dim=1).cpu().detach().item()]

		obj_pred = None; parent_pred=None; obj_count = None
		if (type_pred in self.Types_we_want_obj) or (type_pred in self.Types_we_want_parent):
			text = self.add_task_type_prompting(text, type_pred)
			tok = self.tokenizer(text, truncation=True, padding='max_length')

		if type_pred in self.Types_we_want_obj:
			obj_pred = self.obj_model(torch.tensor([tok['input_ids']]).cuda())
			obj_pred = self.obj_idx_2_obj[torch.argmax(obj_pred, dim=1).cpu().detach().item()]

		if type_pred in self.Types_we_want_parent:
			parent_pred = self.parent_model(torch.tensor([tok['input_ids']]).cuda())
			parent_pred = self.parent_idx_2_parent[torch.argmax(parent_pred, dim=1).cpu().detach().item()]


		return type_pred, obj_count, obj_pred, parent_pred

