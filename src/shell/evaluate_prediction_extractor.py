# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import json
from types import SimpleNamespace

import pickle

import faulthandler; faulthandler.enable()

# Adding path for utils
sys.path.append('../utils')
from emmental_utils import load_data_from_db, get_task

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file','-f',type=str, required=True)
parser.add_argument('--config','-c',type=str,
                    default='/dfs/scratch1/jdunnmon/data/memex-data/config/config.json')
args = parser.parse_args()
args = vars(args)

# Getting config
with open(args['config']) as fl:
    config = json.load(fl)

# Changing directory to code area
os.chdir(config['homedir'])

#For PostgreSQL
if '.' in args['file']:
    filename = os.path.split(args['file'])[-1].split('.')[0]
else:
    filename = os.path.split(args['file'])[-1]

if 'postgres_db_name' not in config.keys():
    postgres_db_name = filename
else:
    postgres_db_name = config['postgres_db_name']

if config['use_pg']:
    print(postgres_db_name)
    os.environ['SNORKELDB'] = os.path.join(config['postgres_location'],
                              postgres_db_name)
else:
    print('Using SQLite...')

# Setting extraction name
extraction_name = 'prediction'

# Loading char dict used for training
print(os.getcwd())
with open(f"{config['prediction_model_path']}/char_dict.pkl",'rb') as fl:
    char_dict = pickle.load(fl)

# Creating dataset
datasets = {}
datasets['test'] = load_data_from_db(postgres_db_name, config['postgres_location'], {}, 
                                     char_dict=char_dict, clobber_label=True)

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

import emmental
from emmental import Meta
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_arg import parse_arg, parse_arg_to_config

# HACK: To get Emmental to initialize correctly
Meta.reset() 

#parser_emm = parse_arg()
#args_emm = parser_emm.parse_args()
#args_emm = vars(args_emm)
#args_emm.update({'model_path': config['prediction_model_path']})
#args_emm = SimpleNamespace(**args_emm)
#config_emm = parse_arg_to_config(args_emm)   
emmental.init(config={'model_path':f"{config['prediction_model_path']}/checkpoint.pth"})

Meta.config["model_config"]["model_path"] = f"{config['prediction_model_path']}/checkpoint.pth"

# Defining tasks
task_names = ["ht_page"]
    
# Defining task to label dict
task_to_label_dict = {"ht_page": "label"}
        
# Getting size of char dict -- assume all chars appear in 1st 1000 examples!
# HACK: HARD CODE THIS/SAVE IT!
#char_dict_size = max([max(datasets['test'].X_dict['emb'][ii]) for ii in range(1000)])+1
 
char_dict_size = char_dict.len()

# Creating dataloaders    
splits = ["test"]
dataloaders = []

for split in splits:
    dataloaders.append(
        EmmentalDataLoader(
            task_to_label_dict={"ht_page": "label"},
            dataset=datasets[split],
            split=split,
            batch_size=16,
            shuffle=False,
            )
        )
    print(f"Built dataloader for {split} set.")

# Getting tasks
tasks = get_task(task_names, config['embed_dim'], char_dict_size)

# Build Emmental model
model = EmmentalModel(name="HT", tasks=tasks)

if Meta.config["model_config"]["model_path"]:
    print('Loading model...')
    model.load(Meta.config["model_config"]["model_path"])

# Scoring
import torch
print("Running prediction model...")
sft = torch.nn.Softmax()
res = model.predict(dataloaders[0], return_preds=True, return_uids=True)
doc_extractions = {}
doc_extractions = {res['uids'][task_names[0]][ii]:{'prediction':str(np.array(sft(torch.Tensor(res['probs'][task_names[0]][ii])))[0])} for ii in range(len(res['uids'][task_names[0]]))}
                      
# Setting filename
out_filename = extraction_name+"_extraction_"+filename+".jsonl"
out_folder = os.path.join(config['output_dir'], extraction_name)
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
print(len(doc_extractions.keys())) 
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    for k,v in doc_extractions.items():
        v['id'] = k
        print(json.dumps(v), file=outfile)
