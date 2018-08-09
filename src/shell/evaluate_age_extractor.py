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
    
# Setting extraction type -- should be a subfield in your data source extractions field!
from dataset_utils import create_candidate_class
extraction_type = 'age'
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
age_ngrams = Ngrams(n_max=3)

# Define matchers
m = RegexMatchSpan(rgx = r'.*(I|He|She) (is|am) ^([0-9]{2})*')
p = RegexMatchSpan(rgx = r'.*(age|is|@|was) ^([0-9]{2})*')
q = RegexMatchSpan(rgx = r'.*(age:) ^([0-9]{2})*')
r = RegexMatchSpan(rgx = r'.*^([0-9]{2}) (yrs|years|year|yr|old|year-old|yr-old|Years|Year|Yr)*')
s = RegexMatchSpan(rgx = r'(^|\W)age\W{0,4}[1-9]\d(\W|$)')

# Union matchers and create candidate extractor
age_matchers = Union(m,p,r,q, s)
cand_extractor = CandidateExtractor(candidate_class, [age_ngrams], [age_matchers])

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
spans = [cand.age.get_span() for cand in eval_cands]
doc_ids = [cand.get_parent().get_parent().name for cand in eval_cands]

# Applying regex
print('Applying filtering regex...')
import re
reg_age = re.compile(r'\d\d')
extractions = [reg_age.search(span).group(0) for span in spans]

# Creating output dictionary
from collections import defaultdict
doc_extractions = defaultdict(list)
for i in range(len(eval_cands)):
    doc_extractions[doc_ids[i]].append(extractions[i])

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
        d[extraction_name] = v[0]
        print(json.dumps(v), file=outfile)