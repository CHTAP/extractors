#Setting Snorkel DB location
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
parser.add_argument('--config','-c',type=str, default='/dfs/scratch1/jdunnmon/data/memex-data/config/config.json')
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
    
# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
#print(f'Number of sentences: {session.query(Sentence).count()}')
print("==============================")

# Getting all documents parsed by Snorkel
print("Getting documents and sentences...")
docs = session.query(Document).all()
#sents = session.query(Sentence).all()

from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import mention_subclass, candidate_subclass
from fonduer.candidates.matchers import RegexMatchSpan, Union

# Defining ngrams for candidates
extraction_name = "age"
age_ngrams = MentionNgrams(n_max=3)

# Define matchers
m = RegexMatchSpan(rgx = r'.*(I|He|She) (is|am) ^([0-9]{2})*')
p = RegexMatchSpan(rgx = r'.*(age|is|@|was) ^([0-9]{2})*')
q = RegexMatchSpan(rgx = r'.*(age:) ^([0-9]{2})*')
r = RegexMatchSpan(rgx = r'.*^([0-9]{2}) (yrs|years|year|yr|old|year-old|yr-old|Years|Year|Yr)*')
s = RegexMatchSpan(rgx = r'(^|\W)age\W{0,4}[1-9]\d(\W|$)')

# Union matchers and create candidate extractor
age_matchers = Union(m,p,r,q, s)

# Getting candidates
AgeMention = mention_subclass("AgeMention")
mention_extractor = MentionExtractor(
        session, [AgeMention], [age_ngrams], [age_matchers]
    )
mention_extractor.clear_all()
mention_extractor.apply(docs, parallelism=parallelism)
candidate_class = candidate_subclass("Age", [AgeMention])
candidate_extractor = CandidateExtractor(session, [candidate_class])

# Applying candidate extractors
candidate_extractor.apply(docs, split=0, parallelism=parallelism)
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
spans = [cand.age_mention.context.get_span() for cand in eval_cands]
doc_ids = [cand.age_mention.document.name for cand in eval_cands]

# Applying regex
print('Applying filtering regex...')
import re
reg_age = re.compile(r'\d\d')
extractions = [reg_age.search(span).group(0) for span in spans]
    
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
out_filename = extraction_name+"_extraction_"+filename+".jsonl"
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
