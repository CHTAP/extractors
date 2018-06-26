from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch_dsl import Search, Q, utils
import csv
import pprint
from requests_aws4auth import AWS4Auth
import os
import boto3 

def pprint_field(fld):
    if type(fld) == utils.AttrDict:
        fld = pprint.pformat(fld.to_dict())
    return fld.encode('utf-8')

AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
region = 'us-east-1' # For example, us-east-1
service = 'es'
host = "https://search-chtap-4-6fit6undgq3aw2r3k6yxcibyjy.us-east-1.es.amazonaws.com"

if len(AWS_ACCESS_KEY) == 0:
    print("Error: Environment variable for AWS_ACCESS_KEY not set.")
    sys.exit()
if len(AWS_SECRET_KEY) == 0:
    print("Error: Environment variable for AWS_SECRET_KEY not set.")
    sys.exit()
if len(host) == 0:
    print("Error: Environment variable for AWS_ES_HOST not set.")
    sys.exit()

awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, region, service)

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

import ipdb; ipdb.set_trace()
index = 'chtap'

max_docs = 1000

q = Q('bool',must=[
      Q("exists",field="memex.extracted_text"),
   #   Q("exists",field="content.extractions.phone"),
      Q("exists",field="content.extraction.location")
      #Q("query_string",**{"default_field":"content.extractions", "query":"*location*"}),
      #Q("nested", path="memex", query=Q("exists",field="memex.extractions"))
      ])

s = Search(using=client, index="chtap").query(q)

# Getting only a certain number of examples...weird way this package does this
s = s[0:max_docs]

res = s.execute()

print("%d documents found" % res['hits']['total'])
print("%d hits found" % len(res['hits']['hits']))

with open('outputfile.tsv', 'w') as csvfile:   
    filewriter = csv.writer(csvfile, delimiter='\t',  # we use TAB delimited, to handle cases where freeform text may have a comma
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    memex_fields = ["id", "content_type" "crawl_data", "crawler", "doc_type", "extracted_metadata", "extracted_text", "extractions", "raw_content", "team", "timestamp", "type", "url", "version"]
    
    content_fields = ["domain", "type", "url", "content", "extractions"]
    
    field_names = [f'memex_{a}' for a in memex_fields]+[a for a in content_fields]
    # create column header row
    filewriter.writerow(field_names)    #change the column labels here
    
    for hit in res['hits']['hits']:
        row = []
        for field in memex_fields:
            try:
                row.append(pprint_field(hit['_source']['memex'][field]))
            except:
                row.append('-1')
        for field in content_fields:
            try:
                row.append(pprint_field(hit['_source']['content'][field]))
            except:
                row.append('-1')

        filewriter.writerow(row)
