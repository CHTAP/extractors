import os
import numpy as np
import findspark
findspark.init()
import sys
sys.path.append('../utils')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
import time
from functools import partial, wraps

    
# Initialize Spark Environment and Spark SQL
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, concat_ws
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf

# Importing utilities
from dataset_utils import retrieve_all_files, get_posting_html_fast, parse_url

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    print('')
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]))
        print('Execution time max: %.3f, average: %.3f' % (max_time, avg_time))
    print('')

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
    

@profile
def read_data_and_drop_na(data_path):
    path_lst = retrieve_all_files(data_path)
    df = spark.read.json(path_lst)
    
    # Dropping samples with no relevant data
    df = df.na.drop(subset=['doc_id','raw_content'])
    return df

@profile
def count_records(df):
    cnt = df.count()
    return cnt 

@profile
def adjust_columns(df, cols):
    # Parsing content and url fields
    df = df.withColumn("raw_content_parsed", get_posting_html_fast_udf(df.raw_content))
    df = df.withColumn("url_parsed", parse_url_udf(df.url))

    # Renaming columns
    df = df.select(attr_list+['raw_content_parsed', 'url_parsed'])
    cols = cols +['raw_content_parsed', 'url_parsed']
    df = df.toDF(*cols)

    # Converting array columns to strings
    for col in cols:
        if 'extracted' in col:
            df = df.withColumn(col, concat_ws(',',col))
            
    return df, cols

@profile
def dump_df_to_csv(df, cols_to_write):
    df.select(cols_to_write).write.csv(write_path,header = 'true',sep='\t',mode='overwrite')

if __name__=="__main__":
    
    # Starting spark session
    spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Snorkel MEMEX Preprocessing") \
    .config("spark.cores.max", "96") \
    .getOrCreate()

    # Defining paths
    data_path = '/lfs/local/0/jdunnmon/data/memex-data/escorts/2016'
    write_path = '/lfs/local/0/jdunnmon/data/memex-data/escorts_preproc/spark_test/2016'

    # Getting list of attributes to extract
    attr_list = ['doc_id','type', 'raw_content','url',\
                 'extractions.phonenumber.results',
                 'extractions.age.results',
                 'extractions.rate.results',
                 #'extractions.location.results',
                 'extractions.ethnicity.results',
                 'extractions.email.results',
                 'extractions.incall.results'
                ]

    # New column names
    cols = ['doc_id','type', 'raw_content','url','extracted_phone','extracted_age',
            'extracted_rate','extracted_ethnicity',\
            'extracted_email','extracted_incall']

    print('Reading dataframe and dropping invalid entries...')
    df = read_data_and_drop_na(data_path)
    print_prof_data()

    # Getting total number of records
    print('Counting records')
    count = count_records(df)
    print(f'{count} records found...')
    print_prof_data()

    # Transforming raw content column
    term = r'([Ll]ocation:[\w\W]{1,200}</.{0,20}>|\W[cC]ity:[\w\W]{1,200}</.{0,20}>|\d\dyo\W|\d\d.{0,10}\Wyo\W|\d\d.{0,10}\Wold\W|\d\d.{0,10}\Wyoung\W|\Wage\W.{0,10}\d\d)'
    get_posting_html_fast_udf = udf(partial(get_posting_html_fast, search_term=term), StringType())
    parse_url_udf = udf(parse_url, StringType())

    # Adding python file to spark context
    spark.sparkContext.addPyFile('../utils/dataset_utils.py')

    # Adjusgint columns
    print('Adjusting columns...')
    df, cols = adjust_columns(df, cols)
    print_prof_data()

    # Writing to csv
    cols_to_write = ['doc_id','type', 'url','url_parsed','raw_content_parsed','extracted_phone','extracted_age',
            'extracted_rate','extracted_ethnicity',\
            'extracted_email','extracted_incall']

    dump_df_to_csv(df, cols_to_write)
    print_prof_data()


