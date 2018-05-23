import codecs
import ast

from builtins import range
import csv
import pycountry
import us
import editdistance

import random
import numpy as np
import re

from snorkel.parser import DocPreprocessor
from snorkel.models import Document, StableLabel
from snorkel.utils import ProgressBar

from snorkel.db_helpers import reload_annotator_labels

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
    dict_string = can.get_parent().document.meta[extractions_field].replace('\n','').strip('"').replace('""',"'")
    extraction = ast.literal_eval(dict_string)[quantity]
    return extraction

def get_gold_labels_from_meta(session, candidate_class, target, split, annotator='gold'):
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split).all()
    
    # Getting all candidates from dev/test set only (splits 1 and 2)
    candidates = session.query(candidate_class).filter(candidate_class.split == split).all()
    cand_total = len(candidates)
    print('Loading', cand_total, 'candidate labels')
    pb = ProgressBar(cand_total)
    
    # Tracking number of labels
    labels=0
    
    # For each candidate, add appropriate gold label
    for i, c in enumerate(candidates):
        pb.bar(i)
        # Get document name for candidate
        stable_id = c.get_parent().stable_id
        # Get text span for candidate
        ext = getattr(c,target)
        val = list(filter(None, re.split('[,/:\s]',ext.get_span().lower())))
        # Get location label from labeled dataframe (input)
        
        try:
            target_strings = get_extraction_from_candidate(c,target,extractions_field='extractions')
        except:
            print('Gold label not found!')
            continue
        # Handling location extraction
        if target == 'location':
                if target_strings == []:
                    targets = ''
                elif type(target_strings) == list :
                    targets = [target.lower() for target in targets]
                elif type(target_strings) == str:
                    targets = list(filter(None,re.split('[,/:\s]',target_strings.lower())))
                if match_val_targets_location(val,targets):
                    label = 1
                else:
                    label = -1
                               
        query = session.query(StableLabel).filter(
            StableLabel.context_stable_ids == stable_id)
        query = query.filter(StableLabel.annotator_name == annotator) 
               
        if query.count() == 0:           
            # Matching target label string to extract span, adding TRUE label if found, FALSE if not
            # This conditional could be improved (use regex, etc.)
            session.add(StableLabel(
                context_stable_ids=stable_id,
                annotator_name=annotator,
                value=label
            ))
            labels+=1

    pb.close()
    print("AnnotatorLabels created: %s" % (labels,))
    
    # Commit session
    session.commit()
    
    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator,
                            split=split, filter_label_split=False)

    
    
#### UTILS FOR CHECKING MATCHES WITH GOLD LABELS ####

def lookup_country_name(cn):
    try:
        out = pycountry.countries.lookup(cn).name
    except:
        out = 'no country'
    return out

def lookup_country_alpha3(cn):
    try:
        out = pycountry.countries.lookup(cn).alpha_3
    except:
        out = 'no country'
    return out

def lookup_country_alpha2(cn):
    try:
        out = pycountry.countries.lookup(cn).alpha_2
    except:
        out = 'no country'
    return out

def lookup_state_name(cn):
    try:
        out = us.states.lookup(val).name
    except:
        out = 'no state'
    return out

def lookup_state_abbr(cn):
    try:
        out = us.states.lookup(val).abbr
    except:
        out = 'no state'
    return out

def check_editdistance(val,targets):
    for tgt in targets:
        if editdistance.eval(val,tgt)<2:
            return True
    return False


def match_val_targets_location(val,targets):
    if val in targets: return True
    if lookup_country_name(val).lower() in targets: return True
    if lookup_country_alpha2(val).lower() in targets: return True
    if lookup_country_alpha3(val).lower() in targets: return True
    if lookup_state_name(val).lower() in targets: return True
    if lookup_state_abbr(val).lower() in targets: return True
    if any([a in val for a in targets]): return True
    if check_editdistance(val,targets): return True
    return False



#def load_external_labels(session, candidate_class, split, preprocessor, annotator='gold',
#    label_fname='data/cdr_relations_gold.pkl', id_fname='data/doc_ids.pkl'):
#    # Load document-level relation annotations
#    with open(label_fname, 'rb') as f:
#        relations = load(f)
#    # Get split candidates
#    candidates = session.query(candidate_class).filter(
#        candidate_class.split == split
#    ).all()
#    for c in candidates:
#        # Get the label by mapping document annotations to mentions
#        doc_relations = relations.get(c.get_parent().get_parent().name, set())
#        label = 2 * int(c.get_cids() in doc_relations) - 1        
#        # Get stable ids and check to see if label already exits
#        context_stable_ids = '~~'.join(x.get_stable_id() for x in c)
#        query = session.query(StableLabel).filter(
#            StableLabel.context_stable_ids == context_stable_ids
#        )
#        query = query.filter(StableLabel.annotator_name == annotator)
#        # If does not already exist, add label
#        if query.count() == 0:
#            session.add(StableLabel(
#                context_stable_ids=context_stable_ids,
#                annotator_name=annotator,
#                value=label
#            ))

#    # Commit session
#    session.commit()

    # Reload annotator labels
#    reload_annotator_labels(session, candidate_class, annotator,
#                            split=split, filter_label_split=False)

# Create unique data file
#import pandas as pd
#df_labeled = pd.read_csv(file_path_subset_unique,sep='\t',names=['ind','source','type','url','content','extractions'])
#df_labeled_unique = df_labeled.drop_duplicates(subset=['url']).reset_index(drop=True)
#df_labeled_unique.to_csv(file_path_unique,sep='\t')