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

# Setting extraction type -- should be a subfield in your data source extractions field!
extraction_name = 'count'

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
print("==============================")

# Executing query for eval candidates
eval_cands = session.query(Document).all()
print(f'Loaded {len(eval_cands)} candidate documents...')

doc_extractions = {'count':len(eval_cands)}
# Setting filename
out_filename = extraction_name+"_extraction_"+filename+".jsonl"
out_folder = os.path.join(config['output_dir'], extraction_name)
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
                          
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    print(json.dumps(doc_extractions), file=outfile)
