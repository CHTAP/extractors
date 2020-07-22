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
extraction_type = 'date'
extraction_name = extraction_type

# Printing number of docs/sentences
from fonduer.parser.models import Document, Sentence
print("==============================")
print(f"DB contents for {postgres_db_name}:")
print(f'Number of documents: {session.query(Document).count()}')
print("==============================")

# Executing query for eval candidates
eval_cands = session.query(Document).all()
print(f'Loaded {len(eval_cands)} candidate documents...')

# Getting gold label for each doc
print("Running regex extractor...")
doc_extractions = {}
for ii, doc in enumerate(eval_cands):
    doc_extractions[doc.name] = {}
    if ii % 1000 == 0:
        print(f'Extracting regexes from doc {ii} out of {len(eval_cands)}')
    doc_extractions[doc.name][extraction_name] = doc.meta['time'].split('T')[0]

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
