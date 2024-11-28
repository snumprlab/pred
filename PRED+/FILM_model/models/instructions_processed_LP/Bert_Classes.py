#BERT CLASSES

import torch

from transformers import BertModel, BertConfig
from torch import nn
import os
import pickle
teach_dir = os.environ['TEACH_DIR']
pickle_dir = os.path.join(teach_dir,os.environ["FILM_model_dir"],'models/instructions_processed_LP', 'BERT_models/bert_pickles')


task_type_to_idx = pickle.load(open(os.path.join(pickle_dir , 'task_type_to_idx.p'), 'rb'))
obj_count_to_idx = pickle.load(open(os.path.join(pickle_dir , 'obj_count_to_idx.p'), 'rb'))
obj_target_to_idx = pickle.load(open(os.path.join(pickle_dir, 'obj_target_to_idx.p'), 'rb'))
parent_target_to_idx = pickle.load( open(os.path.join(pickle_dir, 'parent_target_to_idx.p'), 'rb'))


task_idx_2_type = {v:k for k,v in task_type_to_idx.items()} #This len is 12
obj_idx_2_obj = {v:k for k,v in obj_target_to_idx.items()}
parent_idx_2_parent = {v:k for k,v in parent_target_to_idx.items()}


def get_mlp(in_feature, hid_feature, out_feature):
    return nn.Sequential(
        nn.Linear(in_feature, hid_feature),
        nn.ReLU(),
        nn.Linear(hid_feature, out_feature),
    )

#  or instantiate yourself
# config = BertConfig(
#     vocab_size=2048,
#     max_position_embeddings=768,
#     intermediate_size=2048,
#     hidden_size=512,
#     num_attention_heads=8,
#     num_hidden_layers=6,
#     type_vocab_size=5,
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
#     num_labels=len(task_idx_2_type),
# )

config = BertConfig.from_pretrained('/tmp/bert-base-uncased__config_I_downloaded')

class TaskPredictionType(nn.Module):
	def __init__(self):
		super(TaskPredictionType, self).__init__()
		#self.encoder = BertModel.from_pretrained("/tmp/bert-base-uncased__model_weight_I_downloaded", num_labels=len(task_idx_2_type))
		#self.encoder = BertModel.from_pretrained("bert-base-uncased", num_labels=len(task_idx_2_type))
		self.encoder = BertModel(config)
		# for p in self.encoder.parameters():
		#     p.requires_grad = False
		self.classifier_1 = get_mlp(768, 256, len(task_idx_2_type))
	def forward(self, batch):
		encoded_input = self.encoder(batch).pooler_output
		task_pred = self.classifier_1(encoded_input)
		return task_pred
	
class TaskPredictionObjTarget(nn.Module):
	def __init__(self):
		super(TaskPredictionObjTarget, self).__init__()
		#self.encoder = BertModel.from_pretrained("/tmp/bert-base-uncased__model_weight_I_downloaded", num_labels=len(task_idx_2_type))
		self.encoder = BertModel(config)
		# for p in self.encoder.parameters():
		#     p.requires_grad = False
		self.classifier_3 = get_mlp(768, 256, len(obj_target_to_idx))
	def forward(self, batch):
		encoded_input = self.encoder(batch).pooler_output
		obj_target_pred = self.classifier_3(encoded_input)
		return obj_target_pred
	
class TaskPredictionParentTarget(nn.Module):
	def __init__(self):
		super(TaskPredictionParentTarget, self).__init__()
		#self.encoder = BertModel.from_pretrained("/tmp/bert-base-uncased__model_weight_I_downloaded", num_labels=len(task_idx_2_type))
		self.encoder = BertModel(config)
		# for p in self.encoder.parameters():
		#     p.requires_grad = False
		self.classifier_4 = get_mlp(768, 256, len(parent_idx_2_parent))
	def forward(self, batch):
		encoded_input = self.encoder(batch).pooler_output
		parent_target_pred = self.classifier_4(encoded_input)
		return  parent_target_pred
