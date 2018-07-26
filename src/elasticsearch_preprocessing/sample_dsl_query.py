import os
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import Search, Q, utils
import csv
import pprint
from requests_aws4auth import AWS4Auth

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--extraction_field','-e',type=str,default='all')
parser.add_argument('--index','-i',type=str,default='chtap')
parser.add_argument('--max_docs', '-m', type=int, default=10000)
parser.add_argument('--out_fields', '-of', type=str, default='full')
parser.add_argument('--terms', '-t', type=str, default='')
args = parser.parse_args()

def pprint_field(fld, str_format=True):
    """
    Printing extraction field in formatted string 
    string fld: raw text from field
    bool str_format: encode with utf-8
    """
    if type(fld) == utils.AttrDict:
        fld = pprint.pformat(fld.to_dict())
    if str_format:
        return fld.encode('utf-8')
    else:
        return fld

# Getting environment 
AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
region = 'us-east-1' # For example, us-east-1
service = 'es'
host = os.environ["AWS_HOST"]

if len(AWS_ACCESS_KEY) == 0:
    print("Error: Environment variable for AWS_ACCESS_KEY not set.")
    sys.exit()
if len(AWS_SECRET_KEY) == 0:
    print("Error: Environment variable for AWS_SECRET_KEY not set.")
    sys.exit()
if len(host) == 0:
    print("Error: Environment variable for AWS_ES_HOST not set.")
    sys.exit()

# Creating authorization object
awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, region, service)

# Creating Elasticsearch client

# Use this for IP permissions
#client = Elasticsearch(host, timeout=600)

client = Elasticsearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        http_compress = True,
        timeout = 60,
        request_timeout=30,
        dead_timeout=60,
        retry_on_timeout=True,
        connection_class = RequestsHttpConnection
        )

# Setting index and max docs
index = args.index
max_docs = args.max_docs

# Setting extracton field to query
extraction_field = f'content.extractions.{args.extraction_field}'
terms = args.terms

# Creating query structure
# NOTE: using should instead of must gives many more results!
if args.extraction_field != 'all':
    q = Q('bool',must=[
      Q("exists",field="memex.extracted_text"),
      Q("exists",field=extraction_field)
      ])
elif args.terms:
    q = Q('bool',must=[
      Q("exists",field="memex.extracted_text"),
      Q("match",memex__extracted_text=terms)
      ])
else:
    q = Q('bool',must=[
      Q("exists",field="memex.extracted_text")
      ])



# Executing search
s = Search(using=client, index="chtap").query(q)
res = s.execute()

print("%d documents found" % res['hits']['total'])

# Writing results to file
with open(f'output_{args.extraction_field}.tsv', 'w') as csvfile:   
    filewriter = csv.writer(csvfile, delimiter='\t',  # we use TAB delimited, to handle cases where freeform text may have a comma
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # Setting fields to export    
    id_fields = ["id", "uuid"]
    if args.out_fields == 'full':
        memex_fields = ["id", "content_type", "crawl_data", "crawler", \
                    "doc_type", "extracted_metadata", "extracted_text",\
                    "extractions", "raw_content", "team", "timestamp",\
                     "type", "url", "version"]
        content_fields = ["domain", "type", "url", "content", "extractions"]
    elif args.out_fields == 'extr_text':
        memex_fields = ["id", "extracted_text", "url"]
        content_fields = ["domain", "type", "url", "extractions"]
    else:
        raise ValueError('Invalid output field arg!')
    
    field_names = [f'memex_{a}' for a in memex_fields]+[a for a in content_fields]
    # create column header row
    filewriter.writerow(field_names)    #change the column labels here
   
    for ii,hit in enumerate(s.scan()):
        row = []
        for field in id_fields:
            try:
                row.append(pprint_field(hit[field]))
            except:
                row.append('-1')
        for field in memex_fields:
            try:
                row.append(pprint_field(hit['memex'][field]))
            except:
                row.append('-1')
        for field in content_fields:
            try:
                row.append(pprint_field(hit['content'][field]))
            except:
                row.append('-1')

        filewriter.writerow(row)
        if ii == max_docs:
            break 
