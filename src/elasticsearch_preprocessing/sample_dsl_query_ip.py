from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, utils
import csv
import pprint

def pprint_field(fld):
    if type(fld) == utils.AttrDict:
        fld = pprint.pformat(fld.to_dict())
    return fld.encode('utf-8')

client = Elasticsearch("https://search-chtap-4-6fit6undgq3aw2r3k6yxcibyjy.us-east-1.es.amazonaws.com", timeout=600)

index = 'chtap'

max_docs = 100000

q = Q('bool',must=[
      Q("exists",field="memex.extracted_text"),
      #Q("exists",field="content.extractions.phone"),
      Q("exists",field="content.extractions.location")
      #Q("query_string",**{"default_field":"content.extractions", "query":"*location*"}),
      #Q("nested", path="memex", query=Q("exists",field="memex.extractions"))
      ])

s = Search(using=client, index="chtap").query(q)

# Getting only a certain number of examples...weird way this package does this
#s = s[0:max_docs]

res = s.execute()

print("%d documents found" % res['hits']['total'])
#print("%d hits found" % len(res['hits']['hits']))

with open('es_locations.tsv', 'w') as csvfile:   
    filewriter = csv.writer(csvfile, delimiter='\t',  # we use TAB delimited, to handle cases where freeform text may have a comma
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    id_fields = ["id", "uuid"]
    memex_fields = ["id", "content_type", "crawl_data", "crawler", "doc_type", "extracted_metadata", "extracted_text", "extractions", "raw_content", "team", "timestamp", "type", "url", "version"]
    
    content_fields = ["domain", "type", "url", "content", "extractions"]
    
    field_names = [a for a in id_fields]+[f'memex_{a}' for a in memex_fields]+[a for a in content_fields]
    # create column header row
    filewriter.writerow(field_names)    #change the column labels here
    
   # for hit in res['hits']['hits']:
   # Creating csv -- scan method will access all results of search!
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
