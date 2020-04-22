# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import json

import faulthandler; faulthandler.enable()

# Adding path for utils
sys.path.append('../utils')

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

# Start Snorkel session
#from snorkel import SnorkelSession
#session = SnorkelSession()

from fonduer import Meta
# Start DB connection
conn_string = os.path.join(config['postgres_location'],config['postgres_db_name'])
session = Meta.init(conn_string).Session()

#import torch first to stop TLS error
from dm_utils import LSTM
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

from fonduer.candidates.models import mention_subclass, candidate_subclass
    
# Setting extraction type -- should be a subfield in your data source extractions field!
extraction_name = 'location'
LocationMention = mention_subclass("LocationMention")
candidate_class = candidate_subclass("Location", [LocationMention])

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == 0).count())
print("==============================")

# Split to pull eval candidates from
eval_split = 0

# Executing query for eval candidates
eval_cands = session.query(candidate_class).filter(candidate_class.split == eval_split).order_by(candidate_class.id).all()
print(f'Loaded {len(eval_cands)} candidates...')

# defining model
from dm_utils import LSTM
lstm = LSTM(n_threads=config['dm_parallelism'])

# defining saved weights directory and name
model_name = 'location_lstm' # this was provided when the model was saved!
save_dir = '../../model_checkpoints/location' # this was provided when the model was saved!

# loading
print("Loading LSTM...")
lstm.load(model_name=model_name, save_dir=save_dir, verbose=True, map_location={'cuda:0': 'cpu'}, update_kwargs={'host_device':'cpu', 'use_cuda':False})
lstm.model.use_cuda=False

# Making sure we have a GPU accessible
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Evaluating LSTM
print("Evaluating marginals...")
import torch
if torch.cuda.is_available():
    print(f'Using GPU for {postgres_db_name}')
    #eval_marginals = lstm._marginals_batch(eval_cands)
else:
    print(f'Using CPU for {postgres_db_name}')
    #eval_marginals = lstm.marginals(eval_cands)

#try:
    eval_marginals = lstm._marginals_batch(eval_cands)
    print('Marginals computed...')
#except:
#    print('Exception in marginals')

#eval_marginals = torch.zeros((len(eval_cands),))
# Geocoding
print('Importing gm_utils...')
from gm_utils import create_extractions_dict, create_extractions_dict_parallel
    # Enter googlemaps api key to get geocodes, leave blank to just use extracted locations
geocode_key = None
# geocode_key = 'AIzaSyBlLyOaasYMgMxFGUh2jJyxIG0_pZFF_jM'
print("Creating extractions dictionary...")
doc_extractions = create_extractions_dict_parallel(session, eval_cands, eval_marginals, extractions=[extraction_name],
                                         dummy=False, geocode_key=geocode_key, slices=1)
#except:
#    print('Exception in creating extractions dict')


# Setting filename
out_filename = extraction_name+"_extraction_"+filename+".jsonl"
out_folder = os.path.join(config['output_dir'], extraction_name)
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
                          
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    for k,v in doc_extractions.items():
        v['id'] = k
        print(json.dumps(v), file=outfile)
