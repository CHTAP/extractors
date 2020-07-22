# Create database from preprocessed tsv file

# Setting Snorkel DB location
import os
import sys
import random
import numpy as np
import json

# Adding path for utils
sys.path.append('../utils')

from dataset_utils import set_preprocessor
from snorkel.parser import CorpusParser
from parser_utils import SimpleTokenizer
from snorkel.models import Document, Sentence

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

print(f"Env: {os.environ['SNORKELDB']}")

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

# Setting parallelism
parallelism = config['parallelism']

# Setting random seed
seed = config['seed']
random.seed(seed)
np.random.seed(seed)

# Set data source: options are 'content.tsv', 'memex_jsons', 'es'
data_source = config['data_source']

# Setting max number of docs to ingest
max_docs = config['max_docs']

# Setting location of data source
data_loc = args['file']

# If memex_raw_content is a content_field, uses term as a regex in raw data in addition to getting title and body
#term = r'\b[Ll]ocation:|\b[cC]ity:'
#term = r'(\$?\d\d\d?.*?per|\$?\d\d\d?.*?hours?|\$?\d\d\d?.*?half|\$?\d\d\d?.*?minutes?)'
term = r'([Ll]ocation:.{0,100}|[cC]ity:.{0,100}|\d\dyo\W|\d\d.{0,10}\Wyo\W|\d\d.{0,10}\Wold\W|\d\d.{0,10}\Wyoung\W|\Wage\W.{0,10}\d\d)'

# Doc length in characters, remove to have no max
max_doc_length=None

# Setting preprocessor
print(f'Preprocessing folder: {data_loc}')
doc_preprocessor = set_preprocessor(data_source, data_loc, max_docs=max_docs, verbose=False, clean_docs=False,
                                    content_fields=['raw_content', 'url'], term=term, max_doc_length=max_doc_length)

# Setting parser and applying corpus preprocessor
parser=SimpleTokenizer(delim='<|>')
corpus_parser = CorpusParser(parser=parser)
corpus_parser.apply(list(doc_preprocessor), parallelism=parallelism, verbose=False)

# Printing number of docs/sentences
print("==============================")
print(f"DB creation results for {postgres_db_name}:")
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
print("==============================")
