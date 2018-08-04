import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--threads','-t',type=int, default=1)
parser.add_argument('--data_loc','-d',type=str, required=True)
args = parser.parse_args()

# Getting config
with open('/dfs/scratch1/jdunnmon/data/memex-data/config/config.json') as fl:
    config = json.load(fl)

# Changing directory to code area
os.chdir(config.homedir)

#For PostgreSQL
os.environ['SNORKELDB'] = os.path.join(config.postgres_location,
                              config.postgres_db_name)

# Adding path for utils
sys.path.append('../utils')

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

# Setting parallelism
parallelism = config.parallelism

# Setting random seed
seed = config.seed
random.seed(seed)
np.random.seed(seed)

# If memex_raw_content is a content_field, uses term as a regex in raw data in addition to getting title and body
term = r'\b[Ll]ocation:|\b[cC]ity:'

# Getting raw_content column
print('Getting correct column...')
import pandas as pd
files = os.listdir(args.data_loc)
df = pd.head(files[0])
col = df.columns.get_loc("raw_content")

# Parsing files in parallel
from dataset_utils import parallel_parse_html
print('Preprocessing in parallel...')
parallel_parse_html(args.data_loc, term=term, threads=args.threads, col=col)
