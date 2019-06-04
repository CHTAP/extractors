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

# Setting up parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
# Printing number of docs/sentences
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print("Documents:", session.query(Document).count())
#print("Sentences:", session.query(Sentence).count())
print("==============================")

# Getting all documents parsed by Snorkel
print("Getting documents and sentences...")
docs = session.query(Document).all()
#sents = session.query(Sentence).all()

from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import mention_subclass, candidate_subclass
from dataset_utils import LocationMatcher, city_index
from fonduer.candidates.matchers import Union, LambdaFunctionMatcher
    
# Defining ngrams for candidates
extraction_name = 'location'
ngrams = MentionNgrams(n_max=3)

# Define matchers
cities = city_index('../utils/data/cities15000.txt')
geo_location_matcher = LambdaFunctionMatcher(func=cities.fast_loc)
#spacy_location_matcher = LocationMatcher(longest_match_only=True)
matchers = Union(geo_location_matcher)

# Union matchers and create candidate extractor
print("Extracting candidates...")
LocationMention = mention_subclass("LocationMention")
mention_extractor = MentionExtractor(
        session, [LocationMention], [ngrams], [matchers]
    )
mention_extractor.clear_all()
mention_extractor.apply(docs, parallelism=parallelism)
candidate_class = candidate_subclass("Location", [LocationMention])
candidate_extractor = CandidateExtractor(session, [candidate_class])

# Applying candidate extractors
candidate_extractor.apply(docs, split=0, parallelism=parallelism)
print("==============================")
print(f"Candidate extraction results for {postgres_db_name}:")
print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == 0).count())
print("==============================")
