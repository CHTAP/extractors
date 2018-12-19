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

print(postgres_db_name)
#os.environ['SNORKELDB'] = os.path.join(config['postgres_location'],
#                              postgres_db_name)

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

# Parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
    
# Setting extraction type -- should be a subfield in your data source extractions field!
from dataset_utils import create_candidate_class
extraction_type = 'call'
extraction_name = extraction_type
    
# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Printing number of docs/sentences
from snorkel.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
print(f'Number of sentences: {session.query(Sentence).count()}')
print("==============================")

# Getting all documents parsed by Snorkel
print("Getting documents and sentences...")
docs = session.query(Document).all()
sents = session.query(Sentence).all()

from snorkel.candidates import Ngrams
from snorkel.candidates import CandidateExtractor
from dataset_utils import create_candidate_class
from snorkel.matchers import RegexMatchSpan, Union

# Defining ngrams for candidates
ngrams = Ngrams(n_max=1)

# Define matchers
regex_matcher_1=RegexMatchSpan(rgx = r'(incalls?|outcalls?|incalls?outcalls?|in calls?|out calls?)')

# Union matchers and create candidate extractor
matcher = regex_matcher_1
cand_extractor = CandidateExtractor(candidate_class, [ngrams], [matcher])

# Applying candidate extractors
cand_extractor.apply(sents, split=0, parallelism=parallelism)
print("==============================")
print(f"Candidate extraction results for {postgres_db_name}:")
print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == 0).count())
print("==============================")

# Split to pull eval candidates from
eval_split = 0

# Loading candidates
eval_cands = session.query(candidate_class).order_by(candidate_class.id).all()
print(f'Loaded {len(eval_cands)} candidates...')

# Getting spans and doc_ids
spans = [cand.call.get_span() for cand in eval_cands]
doc_ids = [cand.get_parent().get_parent().name for cand in eval_cands]

# Applying regex
print('Applying filtering regex...')
extractions = [span for span in spans]
    
# Creating output dictionary
from collections import defaultdict
print("Running regex extractor...")
doc_extractions = {}
for ii, cand in enumerate(eval_cands):
    doc_extractions[doc_ids[ii]] = defaultdict(list)
    if ii % 1000 == 0:
        print(f'Extracting regexes from doc {ii} out of {len(eval_cands)}')
    doc_extractions[doc_ids[ii]][extraction_name].append(extractions[ii])

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
    for k,v in doc_extractions.items():
        d['id'] = k
        d[extraction_name] = v[extraction_name]
        print(json.dumps(d), file=outfile)
