# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import pandas as pd
import json

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

from fonduer import Meta
# Start DB connection
conn_string = os.path.join(config['postgres_location'],config['postgres_db_name'])
session = Meta.init(conn_string).Session()

# Setting parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# load ground truth data
gt_pos_path = './ground_truth/confirmed_ht.csv'
gt_neg_path = './ground_truth/experiment_negatives.csv'
gt_pos_df = pd.read_csv(gt_pos_path)
gt_neg_df = pd.read_csv(gt_neg_path)
GT_POS = set(['+1'+ a for a in gt_pos_df[gt_pos_df['ht'].astype(int)==1]['tip'].astype(str).tolist()])
GT_NEG = set(['+1'+ a for a in gt_neg_df[gt_neg_df['ht'].astype(int)==0]['number'].astype(str).tolist()])

import phonenumbers
def matcher(doc):
    phone_list = []
    for s in doc.sentences:
        txt = s.text
        for match in phonenumbers.PhoneNumberMatcher(txt,"US"):
            format_match = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
            phone_list.append(str(format_match))
    phone_set = set(phone_list)
    if set.intersection(phone_set, GT_POS):
        return [1]
    if set.intersection(phone_set, GT_NEG):
        return [2]
    else:
        return [0]
    
# Setting extraction type -- should be a subfield in your data source extractions field!
from dataset_utils import create_candidate_class
extraction_type = 'ground_truth'
extraction_name = extraction_type

# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
print("==============================")

# Executing query for eval candidates
eval_cands = session.query(Document).all()
print(f'Loaded {len(eval_cands)} candidate documents...')

# Getting gold label for each doc
print("Running regex extractor...")
doc_extractions = {}
for ii, doc in enumerate(eval_cands):
    doc_extractions[doc.name] = {}
    if ii % 1000 == 0:
        print(f'Extracting regexes from doc {ii} out of {len(eval_cands)}')
    doc_extractions[doc.name]['ground_truth'] = matcher(doc)

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
        v['ground_truth'] = list(v['ground_truth'])
        print(json.dumps(v), file=outfile)
