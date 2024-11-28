#Fake script to downlaod the autotokenizer when building docker
import sys
import os
#sys.path.append(os.path.join(os.environ["FILM_model_dir"], 'models/instructions_processed_LP'))
#from get_arguments_from_bert import GetArguments

#a = GetArguments()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dir_name = '/tmp/bert-base-uncased_I_downloaded'

if os.path.isdir(dir_name) == False:
    os.mkdir(dir_name)  

tokenizer.save_pretrained(dir_name)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=200)

dir_name = '/tmp/bert-base-uncased_I_downloaded2'

if os.path.isdir(dir_name) == False:
    os.mkdir(dir_name)  

tokenizer.save_pretrained(dir_name)

#Rename config file now

#tmp = AutoTokenizer.from_pretrained(dir_name)   

from transformers import BertModel, BertConfig
#encoder = BertModel.from_pretrained("bert-base-uncased", num_labels=12)
config = BertConfig.from_pretrained('bert-base-uncased') 

dir_name = '/tmp/bert-base-uncased__config_I_downloaded'


if os.path.isdir(dir_name) == False:
    os.mkdir(dir_name)  

config.save_pretrained(dir_name)

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
#     num_labels=3,
# )

# encoder = BertModel(config)