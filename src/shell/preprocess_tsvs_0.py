import json
import sys, os
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--threads','-t',type=int, default=1)
parser.add_argument('--data_loc','-d',type=str, required=True)
args = parser.parse_args()
args = vars(args)

# Getting config
with open('/dfs/scratch1/jdunnmon/data/memex-data/config/config.json') as fl:
    config = json.load(fl)

# Changing directory to code area
os.chdir(config['homedir'])

# Adding path for utils
sys.path.append('../utils')

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# If memex_raw_content is a content_field, uses term as a regex in raw data in addition to getting title and body
term = r'\b[Ll]ocation:|\b[cC]ity:'

# Getting raw_content column
print('Getting correct column...')
import pandas as pd
files = os.listdir(args['data_loc'])
df = pd.read_csv(os.path.join(args['data_loc'],files[0]),sep='\t',nrows=10)
col = df.columns.get_loc("memex_raw_content")

# Parsing files in parallel
from dataset_utils import parallel_parse_html
print('Preprocessing in parallel...')
parallel_parse_html(args['data_loc'], term=term, threads=args['threads'], col=col)
