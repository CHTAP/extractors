import codecs
import ast

import random
import numpy as np

from snorkel.parser import DocPreprocessor
from snorkel.models import Document

class MemexTSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file with one (doc_name <tab> doc_text) per line"""
    
    def __init__(self, path, encoding="utf-8", max_docs=float('inf')):
        super().__init__(path, encoding=encoding, max_docs=max_docs)
        

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for ind, line in enumerate(tsv):
                if ind == 0:
                    continue
                try:
                    (ind, domain, source,  doc_name, doc_text, extractions) = line.split('\t')
                except:
                    print('Malformatted Line!')
                    continue
                if len(doc_text) < 10:
                    print('Short Doc!')
                    continue
                stable_id = self.get_stable_id(doc_name)
                
                doc = Document(
                    name=doc_name, stable_id=stable_id,
                    meta={'domain': domain,
                          'source': source,
                          'extractions':extractions}
                )
                yield doc, doc_text
                
                
def create_test_train_splits(docs, dev_frac=0.1, test_frac=0.1, seed=123):
    ld   = len(docs)
    dev_set_sz = np.round(ld*dev_frac)
    test_set_sz = np.round(ld*test_frac)
    train_set_sz = ld - dev_set_sz - test_set_sz

    # Setting up train, dev, and test sets
    train_docs = set()
    dev_docs   = set()
    test_docs  = set()
    
    train_sents = set()
    dev_sents = set()
    test_sents = set()

    # Creating list of (document name, document object) tuples
    random.seed(seed)
    random.shuffle(docs)

    # Adding unlabeled data to train set, 
    # labeled data to dev/test sets in alternating fashion
    for i, doc in enumerate(docs):
        if i<train_set_sz:
            train_docs.add(doc)
            for s in doc.sentences:
                train_sents.add(s)
        else:
            if len(dev_docs)<=len(test_docs):
                dev_docs.add(doc)
                for s in doc.sentences:
                    dev_sents.add(s)
            else:
                test_docs.add(doc)
                for s in doc.sentences:
                    test_sents.add(s)
                
    #Printing length of train/test/dev sets
    print(f'Train: {len(train_docs)} Docs, {len(train_sents)} Sentences')
    print(f'Dev: {len(dev_docs)} Docs, {len(dev_sents)} Sentences')
    print(f'Test: {len(test_docs)} Docs, {len(test_sents)} Sentences')
    
    return list(train_docs), list(dev_docs), list(test_docs), \
           list(train_sents), list(dev_sents), list(test_sents)

def get_extraction_from_candidate(can,quantity,extractions_field='extractions'):
    return ast.literal_eval(can.meta[extractions_field])[quantity]


# Create unique data file
#import pandas as pd
#df_labeled = pd.read_csv(file_path_subset_unique,sep='\t',names=['ind','source','type','url','content','extractions'])
#df_labeled_unique = df_labeled.drop_duplicates(subset=['url']).reset_index(drop=True)
#df_labeled_unique.to_csv(file_path_unique,sep='\t')