import argparse
import findspark
import json
import numpy as np
import pandas as pd
import random
import sys, os

# Adding path for utils
sys.path.append('../utils')
from dataset_utils import retrieve_all_files

# Initializing pyspark
findspark.init()
import pyspark

# Importing config file
from config import *
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir','-d',type=str, required=True, help='Parent directory of MEMEX jsonl database')
    parser.add_argument('--out_dir','-o',type=str, required=True, help='Output directory for saving parsed data')
    args = parser.parse_args()
    args = vars(args)
    
    

    # Setting random seed
    random.seed(seed)
    np.random.seed(seed)

    # If memex_raw_content is a content_field, uses term as a regex in raw data in addition to getting title and body
    term = r'([Ll]ocation:[\w\W]{1,200}</.{0,20}>|\W[cC]ity:[\w\W]{1,200}</.{0,20}>|\d\dyo\W|\d\d.{0,10}\Wyo\W|\d\d.{0,10}\Wold\W|\d\d.{0,10}\Wyoung\W|\Wage\W.{0,10}\d\d)'
    
    # Setting field dictionary and raw content field
    field_dict = {'id':'doc_id', 'uuid':'', 'memex_id':'doc_id', 'memex_doc_type':'type', 'memex_raw_content':'raw_content', 'memex_url':'url', 'url':'', 'extractions':'extractions'}
    content_field = 'memex_raw_content'

    # json keys:
    # (['type', 'crawl_data', 'url', 'timestamp', 'extractions', 'raw_content', 'extracted_metadata', 'version', 'extracted_text', 'content_type', 'team', 'doc_id', 'crawler'])
    # Getting path and setting up data
    path = args['data_dir']
    file_list = retrieve_all_files(path)
    print(f'Retrieved {len(file_list)} files...')
    file_data = []
    for in_file in file_list:
        if in_file.endswith('jsonl.gz'):
            in_loc = in_file
            in_parts = in_loc.split('/')
            in_parts[in_parts.index('escorts')] = args['out_dir']
            in_parts[-1] = in_parts[-1].split('.')[0]+'.tsv'
            out_loc = '/'.join(in_parts)
            arg_tuple = (in_loc, out_loc, term, field_dict, content_field)
            file_data.append(arg_tuple)
    #path_list = [os.path.join(path, file) for file in file_list]
    #file_data = [(path, term) for path in path_list if path.endswith('tsv')]

    print('Processing in parallel')

    sc = pyspark.SparkContext(appName="parse_html_from_jsonl_gz")
    sc.addPyFile('../utils/dataset_utils.py')
    from dataset_utils import parse_html_from_jsonl_gz
    print('Distributing data...')
    distData = sc.parallelize(file_data, 100)
    print('Parsing distributed data...')
    distData.foreach(parse_html_from_jsonl_gz)
    sc.stop()
