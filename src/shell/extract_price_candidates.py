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
if 'postgres_db_name' not in config.keys():
    postgres_db_name = os.path.split(args['file'])[-1].split('.')[0]
else:
    postgres_db_name = config['postgres_db_name']

if config['use_pg']:
    print(postgres_db_name)
    os.environ['SNORKELDB'] = os.path.join(config['postgres_location'],
                              postgres_db_name)
else:
    print('Using SQLite...')

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

#import torch first to stop TLS error
from dm_utils import LSTM

# Setting parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# Printing number of docs/sentences
from snorkel.models import Document, Sentence
# Printing number of docs/sentences
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
print("==============================")

# Getting all documents parsed by Snorkel
print("Getting documents and sentences...")
docs = session.query(Document).all()
sents = session.query(Sentence).all()

from snorkel.candidates import Ngrams
from snorkel.candidates import CandidateExtractor
from dataset_utils import create_candidate_class, price_match
from snorkel.matchers import LambdaFunctionMatcher

# Setting extraction type -- should be a subfield in your data source extractions field!
extraction_type = 'price'

# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Defining ngrams for candidates
price_ngrams = Ngrams(n_max=5)

# Define matchers
price_matcher = LambdaFunctionMatcher(func=price_match)

# Union matchers and create candidate extractor
cand_extractor = CandidateExtractor(candidate_class, [price_ngrams], [price_matcher])

# Applying candidate extractors
cand_extractor.apply(sents, split=0, parallelism=parallelism)
print("==============================")
print(f"Candidate extraction results for {postgres_db_name}:")
print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == 0).count())
print("==============================")
