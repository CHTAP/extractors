# Create database from preprocessed tsv file

# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import json
import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file','-f',type=str, required=True)
parser.add_argument('--config','-c',type=str, required=True)
args = parser.parse_args()
args = vars(args)

# Getting config
with open(args['config']) as fl:
    config = json.load(fl)

# Adding path for utils
sys.path.append('../utils')

# Setting parallelism
parallelism = config['parallelism']

# Changing directory to code area
os.chdir(config['homedir'])

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# Loading pre-parsed database
print("Loading pre-parsed DB")
conn_string = os.path.join(config['postgres_location'],config['postgres_db_name'])
cmd = f"psql {conn_string} < {args['file']}"
os.system(cmd)

from fonduer import Meta
# Setting up DB connection
session = Meta.init(conn_string).Session()

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB creation results for {config['postgres_db_name']}:")
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
print("==============================")
