import os
import codecs
import ast
import json
from bs4 import BeautifulSoup
import itertools
import numpy as np

from builtins import range
import csv
import pycountry
import us
import editdistance

import random
import numpy as np
import pandas as pd
import re

from snorkel.parser import DocPreprocessor, HTMLDocPreprocessor
from snorkel.models import Document, StableLabel, GoldLabel, GoldLabelKey
from snorkel.utils import ProgressBar

from snorkel.db_helpers import reload_annotator_labels

import gzip

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
                          
def create_test_train_splits(docs, quantity, gold_dict=None, dev_frac=0.1, test_frac=0.1, seed=123):
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

    # Creating list of document objects
    random.seed(seed)
    random.shuffle(docs)
    
    # Creating list of urls to check against gold dict
    if gold_dict is not None:
        urls = set([doc.name for doc in docs])
        gold_urls = set(list(gold_dict.keys()))
        gold_list = list(set.intersection(urls, gold_urls))

    # Adding unlabeled data to train set, 
    # ensuring gold labeled data is added to test/dev
    for i, doc in enumerate(docs):
        try:
            if gold_dict is None:
                quant_ind = check_extraction_for_doc(doc, quantity, extractions_field='extractions')
            else:
                quant_ind = doc.name in gold_list
        except:
            print('Malformatted JSON Entry!')
        if quant_ind and (len(dev_docs)<dev_set_sz )and (len(dev_docs) < len(test_docs)):
            dev_docs.add(doc)
            for s in doc.sentences:
                dev_sents.add(s)
        elif quant_ind and (len(test_docs)<test_set_sz) :
            test_docs.add(doc)
            for s in doc.sentences:
                test_sents.add(s)        
        elif len(train_docs)<train_set_sz:
            train_docs.add(doc)
            for s in doc.sentences:
                train_sents.add(s)
                
    #Printing length of train/test/dev sets
    print(f'Train: {len(train_docs)} Docs, {len(train_sents)} Sentences')
    print(f'Dev: {len(dev_docs)} Docs, {len(dev_sents)} Sentences')
    print(f'Test: {len(test_docs)} Docs, {len(test_sents)} Sentences')
    
    return list(train_docs), list(dev_docs), list(test_docs), \
           list(train_sents), list(dev_sents), list(test_sents)

def check_extraction_for_doc(doc, quantity, extractions_field='extractions'):
    if quantity is None:
        return True
    dict_string = doc.meta[extractions_field].strip('\n').strip('"').replace('""','"').replace('\\"',"\\").replace('\\','\\\\')
    extraction_dict = json.loads(dict_string)
    if quantity in list(extraction_dict.keys()):
        return True
    return False

def get_extraction_from_candidate(can,quantity,extractions_field='extractions'):
    dict_string = can.get_parent().document.meta[extractions_field].strip('\n').strip('"').replace('""','"').replace('\\"',"\\").replace('\\','\\\\')
    extraction = json.loads(dict_string)[quantity]
    return extraction

def get_candidate_stable_id(can):
    return can.get_parent().stable_id+str(can.id)

def check_gold_perc(session):
    gold_labels = session.query(GoldLabel).all()
    gold_vals = np.array([a.value for a in gold_labels])
    perc_pos = np.sum(gold_vals == 1)/len(gold_vals)
    print(f'Percent Positive: {perc_pos:0.2f}')
    return perc_pos

def get_gold_labels_from_meta(session, candidate_class, target, split, annotator='gold', gold_dict=None):
    
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split).all()
    
    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator).first()
    if ak is None:
        ak = GoldLabelKey(name=annotator)
        session.add(ak)
        session.commit()   
    
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
        # Get stable id for candidate
        stable_id = get_candidate_stable_id(c)
        # Get text span for candidate
        ext = getattr(c,target)
        url = c.get_parent().document.name
        val = list(filter(None, re.split('[,/:\s]',ext.get_span().lower())))
        # Get location label from labeled dataframe (input)
        if gold_dict is None:
            try:
                target_strings = get_extraction_from_candidate(c,target,extractions_field='extractions')
            except:
                print('Gold label not found!')
                continue
        else:
            try:
                target_strings = gold_dict[url]
            except:
                print('Gold label not found!')
                continue
        # Handling location extraction
        if target == 'location':
            if target_strings == []:
                targets = ''
            elif type(target_strings) == list :
                targets = [target.lower() for target in target_strings]
            elif type(target_strings) == str:
                targets = list(filter(None,re.split('[,/:\s]',target_strings.lower())))
                
            targets_split = list(itertools.chain.from_iterable([t.split() for t in targets]))

            if match_val_targets_location(val,targets) or match_val_targets_location(val,targets_split):
                label = 1
            else:
                label = -1
                    
            existing_label = session.query(GoldLabel).filter(GoldLabel.key == ak).filter(GoldLabel.candidate == c).first()
            if existing_label is None:
                session.add(GoldLabel(candidate=c, key=ak, value=label))                        
                labels+=1

    pb.close()
    print("AnnotatorLabels created: %s" % (labels,))
    
    # Commit session
    session.commit()
    
    # Reload annotator labels
    #reload_annotator_labels(session, candidate_class, annotator,
    #                        split=split, filter_label_split=False)

       # query = session.query(StableLabel).filter(
       #     StableLabel.context_stable_ids == stable_id)
       # query = query.filter(StableLabel.annotator_name == annotator) 
               
        #if query.count() == 0:           
            # Matching target label string to extract span, adding TRUE label if found, FALSE if not
            # This conditional could be improved (use regex, etc.)
        #    session.add(StableLabel(
        #        context_stable_ids=stable_id,
        #        annotator_name=annotator,
        #        value=label
        #    ))    
    
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


def remove_gold_labels(session):
    session.query(GoldLabel).delete()
    
def retrieve_all_files(dr):
    """
    Recurively returns all files in root directory
    """
    lst = []
    for root, directories, filenames in os.walk(dr): 
         for filename in filenames: 
                lst.append(os.path.join(root,filename))
    return lst


class HTMLListPreprocessor(HTMLDocPreprocessor):
    
    def __init__(self, path, file_list, encoding="utf-8", max_docs=float('inf')):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.file_list = file_list
        
    def _get_files(self,path_list):
        fpaths = [os.path.join(self.path,fl) for fl in path_list]
        return fpaths
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        for fp in self._get_files(self.file_list):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(fp, file_name):
                    yield doc, text
                    doc_count += 1
                    if doc_count >= self.max_docs:
                        return
                    
class MEMEXJsonLGZIPPreprocessor(HTMLListPreprocessor):
    
    def __init__(self, path, file_list, encoding="utf-8", max_docs=float('inf'), lines_per_entry=6, verbose=False, content_field='raw_content'):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.file_list = file_list
        self.lines_per_entry = lines_per_entry
        self.verbose=verbose
        self.urls = []
        self.content_field = content_field
        
    def _get_files(self,path_list):
        fpaths = [fl for fl in path_list]
        return fpaths
    
    def _can_read(self, fpath):
        return fpath.endswith('jsonl') or fpath.endswith('gz')
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        for file_name in self._get_files(self.file_list):
            if self._can_read(file_name):
                for doc, text in self.parse_file(file_name):
                    yield doc, text
                    doc_count += 1
                    if self.verbose:
                        print(f'Parsed {doc_count} docs...')
                    if doc_count >= self.max_docs:
                        return
                    
    def _lines_per_n(self, f, n):
        for line in f:
            yield ''.join(chain([line], islice(f, n - 1)))
        
    def _read_content_file(self, fl):
        json_lst = []
        if fl.endswith('gz'):
            with gzip.GzipFile(fl, 'r') as fin: 
                f = fin.read()
            for chunk in f.splitlines():
                jfile = json.loads(chunk)
                json_lst.append(jfile)

        elif fl.endswith('jsonl'):
            with open(fl) as f:
                for chunk in self._lines_per_n(f, self.lines_per_entry):
                    jfile = json.loads(chunk)
                    json_lst.append(jfile)
        else:
            print('Unrecognized file type!')
                    
        json_pd = pd.DataFrame(json_lst)
        #json_pd = pd.DataFrame(json_lst).dropna()
        return json_pd
    
    def parse_file(self, file_name):
        df = self._read_content_file(file_name)
        if (self.content_field in df.keys()):
            for index, row in df.iterrows():
                name = row.url
                memex_doc_id = row.doc_id
                content = getattr(row,self.content_field)
                # Added to avoid duplicate keys
                if name in self.urls:
                    continue
                if type(content) == float:
                    continue
                if content is None:
                    continue
                stable_id = self.get_stable_id(memex_doc_id)
                #try:
                
                html = BeautifulSoup(content, 'lxml')
                text = list(filter(self._cleaner, html.findAll(text=True)))
                text = ' '.join(str(self._strip_special(s)) for s in text if s != '\n')
                   #text = ' '.join(row.raw_content[1:-1].replace('<br>', '').split())
                #text = row.raw_content[1:-1].encode(self.encoding)
                self.urls.append(name)
                yield Document(name=name, stable_id=stable_id,
                                       meta={'file_name' : file_name, 'memex_doc_id' : memex_doc_id}), str(text)
                #except:
                #    print('Failed to parse document!')
        else:
            print('File with no raw content!')
            

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