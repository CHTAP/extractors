from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q , utils
import csv
import pprint

def pprint_field(fld):
    if type(fld) == utils.AttrDict:
        fld = pprint.pformat(fld)
    return fld.encode('utf-8')
    #elif type(fld) == str:
    #    return fld.encode('utf-8')

#client = Elasticsearch("https://search-chtap-3-mm7biqczdg2icuw4ldidjsfc3e.us-east-1.es.amazonaws.com")

client = Elasticsearch("https://search-chtap-4-6fit6undgq3aw2r3k6yxcibyjy.us-east-1.es.amazonaws.com")

index = 'chtap'

max_docs = 100
#body = {"query": {"bool":{"must":{"exists":{"field": "memex"}}}}}
#body = {"query": {
#           "bool":{"must": [{"nested": {"path":"memex","query":{
#                "bool":{"must":[{"exists": {"field":"extractions"
#                 }}]}}
#            }}]}}
#       }  

q = Q('bool',must=[
      Q("exists",field="memex"),
      #Q("exists",field="content.extractions.location"),
    #  Q("match","content.extractions"="location"),
#      Q("nested", path="memex", query=Q("exists",field="memex.extractions"))
      ])

s = Search(using=client, index="chtap").query(q)

# Getting only a certain number of examples...weird way this package does this
s = s[0:max_docs]

res = s.execute()

#import ipdb; ipdb.set_trace()
#res = es.search(index="chtap", body=body)
#res = es.search(index="chtap",  body={"query": {"simple_query_string": {"query": "phone","fields":["extractions"]}}})

#dir(res['hits']['hits'][7]['_source']['memex']['extractions'])
print("%d documents found" % res['hits']['total'])
print("%d hits found" % len(res['hits']['hits']))
import ipdb; ipdb.set_trace()

res_red = res['hits']['hits'][0:500]

with open('outputfile.tsv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter='\t',  # we use TAB delimited, to handle cases where freeform text may have a comma
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    memex_fields = ["id", "content_type" "crawl_data", "crawler", "doc_type", "extracted_metadata", "extracted_text", "extractions", "raw_content", "team", "timestamp", "type", "url", "version"]
    content_fields = ["domain", "type", "url", "content", "extractions"]

    field_names = [f'memex_{a}' for a in memex_fields]+[a for a in content_fields]
    # create column header row
    filewriter.writerow(field_names)    #change the column labels here
    #import ipdb; ipdb.set_trace() 
    for hit in res_red:
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
