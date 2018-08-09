# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import json

# Adding path for utils
sys.path.append('../utils')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file','-f',type=str, required=True)
args = parser.parse_args()
args = vars(args)

# Getting config
with open('/dfs/scratch1/jdunnmon/data/memex-data/config/config.json') as fl:
    config = json.load(fl)

# Changing directory to code area
os.chdir(config['homedir'])

#For PostgreSQL
postgres_db_name = os.path.split(args['file'])[-1].split('.')[0]
os.environ['SNORKELDB'] = os.path.join(config['postgres_location'], 
                              postgres_db_name)

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

# Setting parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# Defining LFs
import re
from snorkel.lf_helpers import get_tagged_text, get_left_tokens, get_right_tokens, get_between_tokens
from gm_utils import *

def lf_preceding_half_missing_quantity(c):
    reg_forward = re.compile(r'([^\d]\$?\d\d\d?[^\d,]{0,10}[^a-z0-9,](half|hh)|[^\d]\$?\d\d\d?(half|hh)|[^\d]\$?\d\d\d?[^\d,]{0,10}(30|45).?minutes?)')
    reg_half = re.compile(r'(half|hh)|(30|45).?minutes?')
    sent = c.get_parent().text.lower()

    return -1 if reg_half.search(sent) and not reg_forward.search(sent) else 0

def lf_preceding_half(c):
    preceding_words = ['half', 'minutes']
    return -1 if overlap(
      preceding_words,
      get_left_tokens(c, window=1)) else 1

LFs  =  [
    lf_preceding_half_missing_quantity,
    lf_preceding_half
]    

# Setting extraction type -- should be a subfield in your data source extractions field!
from dataset_utils import create_candidate_class
extraction_type = 'price'

# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Printing number of docs/sentences
from snorkel.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == 0).count())
print("==============================")

# Split to pull eval candidates from
eval_split = 0

# Executing query for eval candidates
eval_cands = session.query(candidate_class).filter(candidate_class.split == eval_split).order_by(candidate_class.id).all()
print(f'Loaded {len(eval_cands)} candidates...')

# Applying LFs
print("Applying LFs...")
from snorkel.annotations import LabelAnnotator
labeler = LabelAnnotator(lfs=LFs)
L_eval = labeler.apply(split=eval_split, parallelism=parallelism)

# defining model
from snorkel.learning import GenerativeModel
# Creating generative model
gen_model = GenerativeModel()

# defining saved weights directory and name
model_name = 'Price_Gen_20K' # this was provided when the model was saved!
save_dir = '/dfs/scratch0/jdunnmon/data/memex-data/extractor_checkpoints/Price_Gen_20K' # this was provided when the model was saved!

# loading
print("Loading generative model...")
gen_model.load(model_name=model_name, save_dir=save_dir, verbose=True)

# Evaluating LSTM
print("Evaluating marginals...")
eval_marginals = gen_model.marginals(L_eval)

# Geocoding
from gm_utils import create_extractions_dict
# Enter googlemaps api key to get geocodes, leave blank to just use extracted locations
geocode_key = None
# geocode_key = 'AIzaSyBlLyOaasYMgMxFGUh2jJyxIG0_pZFF_jM'
print("Creating extractions dictionary...")
doc_extractions = create_extractions_dict(session, L_eval, eval_marginals, extractions=[extraction_type],
                                          dummy=False, geocode_key=geocode_key)

# Setting filename
out_filename = "price_extraction_"+postgres_db_name+".jsonl"
out_folder = os.path.join(config['output_dir'], 'price')
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
                          
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    for k,v in doc_extractions.items():
        v['id'] = k
        print(json.dumps(v), file=outfile)