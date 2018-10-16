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

# Defining regex matcher function
import phonenumbers, re
def regex_matcher(doc, mode=phonenumbers):
    phone_list = []
    results_list = []
    r = re.compile(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}')
    if mode == 'regex':
        for s in doc.sentences:
            txt = s.text
            results = r.findall(txt)
            for x in results:
                phone_list.append(str(x))
    elif mode == 'phonenumbers':
         for s in doc.sentences:
            txt = s.text
            for match in phonenumbers.PhoneNumberMatcher(txt,"US"):
                format_match = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
                phone_list.append(str(format_match))
                
    return list(set(phone_list))
    
# Setting extraction type -- should be a subfield in your data source extractions field!
from dataset_utils import create_candidate_class
extraction_type = 'phone'

# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Printing number of docs/sentences
from snorkel.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
print("==============================")

# Split to pull eval candidates from
eval_split = 0

# Executing query for eval candidates
eval_cands = session.query(Document).all()
print(f'Loaded {len(eval_cands)} candidate documents...')

# defining model
from dm_utils import LSTM
lstm = LSTM(n_threads=parallelism)

# Getting gold label for each doc
print("Running regex extractor...")
doc_extractions = {}
for ii, _ in enumerate(eval_cands):
    doc_extractions[doc.name] = {}
    if ii % 1000 == 0:
        print(f'Extracting regexes from doc {ii} out of {len(eval_cands)}')
    doc_extractions[doc.name]['phone'] = regex_matcher(doc, mode='phonenumbers')

# Setting filename
out_filename = "phone_extraction_"+postgres_db_name+".jsonl"
out_folder = os.path.join(config['output_dir'], 'phone')
out_path = os.path.join(out_folder, out_filename)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
                          
# Saving file to jsonl in extractions format
print(f"Saving output to {out_path}")
with open(out_path, 'w') as outfile:
    for k,v in doc_extractions.items():
        v['id'] = k
        v['phone'] = list(v['phone'])
        print(json.dumps(v), file=outfile)