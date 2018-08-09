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

# Setting extraction name
extraction_name = "raw_content"

# Getting raw_content column
print('Getting correct column...')
import pandas as pd
df = pd.read_csv(args['file']+'.tsv',sep='\t')
content_col = df.columns.get_loc("memex_raw_content")
id_col = df.columns.get_loc("id")

# Setting filename
out_filename = extraction_name+"_extraction_"+postgres_db_name+".jsonl"
out_folder = os.path.join(config['output_dir'], extraction_name)
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
                          
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    d = {}
    for index, row in df.iterrows():
        d['id'] = row['id']
        d[extraction_name] = row['memex_raw_content']
        print(json.dumps(d), file=outfile)