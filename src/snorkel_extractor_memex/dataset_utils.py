from snorkel.matchers import RegexMatchEach
import os
import codecs
import json
from bs4 import BeautifulSoup
import itertools
import numpy as np

import random
import numpy as np
import pandas as pd
import re
import gzip

import pycountry
import us
import editdistance

from snorkel.parser import DocPreprocessor, HTMLDocPreprocessor
from snorkel.models import Document, Candidate, candidate_subclass, GoldLabel, GoldLabelKey
from snorkel.utils import ProgressBar
    
######################################################################################################
##### HELPER FUNCTIONS FOR GOLD LABELING
######################################################################################################

def lookup_country_name(cn):
    """
    Check if string is a country by name
    
    string cn: candidate text 
    """
    try:
        out = pycountry.countries.lookup(cn).name
    except:
        out = 'no country'
    return out

def lookup_country_alpha3(cn):
    """
    Check if string is a country by three-letter abbreviation
    
    string cn: candidate text 
    """
    try:
        out = pycountry.countries.lookup(cn).alpha_3
    except:
        out = 'no country'
    return out

def lookup_country_alpha2(cn):
    """
    Check if string is a country by two-letter abbreviation
    
    string cn: candidate text 
    """
    try:
        out = pycountry.countries.lookup(cn).alpha_2
    except:
        out = 'no country'
    return out

def lookup_state_name(cn):
    """
    Check if string is a state by direct lookup
    
    string cn: candidate text 
    """
    try:
        out = us.states.lookup(val).name
    except:
        out = 'no state'
    return out

def lookup_state_abbr(cn):
    """
    Check if string is a state abbreviation by direct lookup
    
    string cn: candidate text 
    """
    try:
        out = us.states.lookup(val).abbr
    except:
        out = 'no state'
    return out

def check_editdistance(val,targets):
    """
    Check edit_distance between val and targets
    
    string val: candidate text 
    string targets: target text to match
    """
    for tgt in targets:
        if editdistance.eval(val,tgt)<2:
            return True
    return False


def match_val_targets_location(val,targets):
    """
    Match val to targets for location
    
    string val: candidate text 
    string targets: target text to match
    """
    if val in targets: return True
    if lookup_country_name(val).lower() in targets: return True
    if lookup_country_alpha2(val).lower() in targets: return True
    if lookup_country_alpha3(val).lower() in targets: return True
    if lookup_state_name(val).lower() in targets: return True
    if lookup_state_abbr(val).lower() in targets: return True
    if any([a in val for a in targets]): return True
    if check_editdistance(val,targets): return True
    return False

######################################################################################################
##### HELPER FUNCTIONS
######################################################################################################

def retrieve_all_files(dr):
    """
    Recurively returns all files in root directory
    
    string dr: path to directory
    """
    lst = []
    for root, directories, filenames in os.walk(dr): 
         for filename in filenames: 
                lst.append(os.path.join(root,filename))
    return lst

def replace_str_index(text,index=0,replacement=''):
    """
    Replaces a single index 
    
    string text: string to modify
    int index: index of character to replace
    string replacement: string to replace with
    """
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def replace_middle_double_quotes(text):
    """
    Replaces all double quotes in the middle of a sentence to assist in text conditioning to valid json.
    
    string text: string to modify
    """
    indices = [m.start(0) for m in re.finditer(r'[a-zA-Z0-9_.!?]"[a-zA-Z0-9_.!?]', text)]
    indices = indices + [m.start(0) for m in re.finditer(r'[a-zA-Z0-9_.!?]"\s[a-zA-Z0-9_.!?]', text)]
    indices = indices + [m.start(0) for m in re.finditer(r'[a-zA-Z0-9_.!?]\s"[a-zA-Z0-9_.!?]', text)]
    for ii in indices:
        text = replace_str_index(text,ii+1,"'")
    return text

def clean_extracted_text(text):
    """
    Cleans text extracted from html files -- currently works with data from MEMEX crawl
    
    string text: string to modify
    """
    
    # Replacing and stripping special characters
    text =text.replace('\\n','').replace('\'','"').replace('|','').strip('\n').strip('\r').strip('b').strip('"').replace('\\\"','"').replace('""','"')
    
    # Removing extraneous back-to-back double quotes
    text = " ".join(text.split()).replace('" "','')
    
    # Removing special characters
    text = re.sub(r'\\x[a-zA-Z0-9][a-zA-Z0-9]', '',text)
    # Removing internal double quotes
    text = replace_middle_double_quotes(text)
    
    # Removing remaining escapes
    text = re.sub(r'\\',' ',text)
    
    return text

def check_extraction_for_doc(doc, quantity, extractions_field='extractions', strip_end=False):
    """
    Checking if a document has a particular extraction
    
    Document doc: candidate document
    string quantity: extraction
    string extractions_field: tsv field to use as extractions dict
    bool strip_end: strip start and end characters
    """
    if quantity is None:
        return True
    
    # Getting cleaned text
    dict_string = clean_extracted_text(doc.meta['extractions']) 
    
    # Stripping start/end chars
    if strip_end:
        dict_string = dict_string[1:-1]
        
    # String-to-dict
    extraction_dict = json.loads(dict_string)
    
    # Check if quantity is in extractions field
    if quantity in list(extraction_dict.keys()):
        return True
    return False

def get_extraction_from_candidate(can,quantity,extractions_field='extractions'):
    """
    Getting extraction from MEMEX tsv 
    
    Candidate can: candidate to get extraction from
    string quantity: extraction quantity to retrieve
    string extractions_field: field where extractions dictionary is stored
    """
    
    # Getting cleaned string describing extractions fileld
    dict_string = clean_extracted_text(can.get_parent().document.meta['extractions']) 
    
    # String-to-dict, extract quantity of interest
    extraction = json.loads(dict_string)[quantity]
    
    return extraction

def get_candidate_stable_id(can):
    """
    Creating stable ID for each candidate
    """
    return can.get_parent().stable_id+str(can.id)



######################################################################################################
##### CLASSES FOR DOCUMENT PREPROCESSING
######################################################################################################

# MOST OF THESE CLASSES ARE DIFFS OFF OF EXISTING SNORKEL PREPROCESSORS

class LocationMatcher(RegexMatchEach):
    """
    Matches Spans that are the names of locations, as identified by spaCy.
    A convenience class for setting up a RegexMatchEach to match spans
    for which each token was tagged as a location.
    """

    def __init__(self, *children, **kwargs):
        
        kwargs['attrib'] = 'ner_tags'
        kwargs['rgx'] = 'GPE|LOC'
        super(LocationMatcher, self).__init__(*children, **kwargs)

class HTMLListPreprocessor(HTMLDocPreprocessor):
    """
    Parses a list of html files.
    """
    
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
                    
class ESTSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file drawn from Elasticsearch"""
    
    def __init__(self, path, encoding="utf-8", max_docs=float('inf'), verbose=False, clean_docs=False):
        super().__init__(path, encoding=encoding, max_docs=max_docs)
        self.verbose = verbose
        self.clean_docs = clean_docs

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for ind, line in enumerate(tsv):
                if ind == 0:
                    continue
                try:
                    # Loading data -- ignore malformatted entries!
                    # TODO: Make these fields dynamic/drawn from header? Or make field names an option?
                    (doc_id, uuid, memex_id, memex_content_type, crawl_data, memex_crawler, memex_doc_type, memex_extracted_metadata, memex_extracted_text, memex_extractions, memex_raw_content, memex_team, memex_timestamp, memex_type, memex_url, memex_version, domain, content_type, url, content, extractions) = line.split('\t')
                except:
                    print('Malformatted Line!')
                    continue
                doc_text = content
                doc_name = doc_id
                source = content_type
                extractions = extractions
                # Short documents are usually parsing errors...
                if len(doc_text) < 10:
                    if self.verbose:
                        print('Short Doc!')
                    continue
                    
                # Setting stable id
                stable_id = self.get_stable_id(doc_name)
                
                # Cleaning documents if specified
                if self.clean_docs:
                    doc_text = doc_text.replace('\n',' ').replace('\t',' ').replace('<br>', ' ')
                    # Eliminating extra space
                    doc_text = " ".join(doc_text.split())
                
                # Yielding reults, adding useful info to metadata
                doc = Document(
                    name=doc_name, stable_id=stable_id,
                    meta={'domain': domain,
                          'source': source,
                          'extractions':extractions,
                          'url':url}
                )
                yield doc, doc_text
                
class MemexTSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file from MEMEX (content.tsv)"""
    
    def __init__(self, path, encoding="utf-8", max_docs=float('inf'), verbose=False, clean_docs=False):
        super().__init__(path, encoding=encoding, max_docs=max_docs)
        self.verbose = verbose
        self.clean_docs = clean_docs

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
                    if self.verbose:
                        print('Short Doc!')
                    continue
                stable_id = self.get_stable_id(doc_name)
                
                if self.clean_docs:
                    doc_text = doc_text.replace('\n',' ').replace('\t',' ').replace('<br>', ' ')
                    # Eliminating extra space
                    doc_text = " ".join(doc_text.split())
                
                doc = Document(
                    name=doc_name, stable_id=stable_id,
                    meta={'domain': domain,
                          'source': source,
                          'extractions':extractions}
                )
                yield doc, doc_text
                        
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
                if self.content_field == 'raw_content':
                    html = BeautifulSoup(content, 'lxml')
                    text = list(filter(self._cleaner, html.findAll(text=True)))
                    text = ' '.join(str(self._strip_special(s)) for s in text if s != '\n')
                   #text = ' '.join(row.raw_content[1:-1].replace('<br>', '').split())
                #text = row.raw_content[1:-1].encode(self.encoding)
                elif self.content_field == 'extracted_text':
                    # Elimintating extraneous newlines and special characters
                    text = content.replace('\n',' ').replace('\t',' ')
                    # Eliminating extra space
                    text = " ".join(text.split())
                else:
                    raise ValueError('Invalid content field!')
                
                self.urls.append(name)
                yield Document(name=name, stable_id=stable_id,
                                       meta={'file_name' : file_name, 'memex_doc_id' : memex_doc_id}), str(text)
                #except:
                #    print('Failed to parse document!')
        else:
            print('File with no raw content!')

######################################################################################################
##### EXPOSED FUNCTIONS
######################################################################################################

def set_preprocessor(data_source,data_loc,max_docs=1000,verbose=False,clean_docs=True,content_field='extracted_text'): 
    """
    Sets a chosen document preprocessor.
    
    string data_source: type of data soure -- 'content.tsv', 'es' (Elasticsearch), 'memex_jsons'
    string data_loc: path to data source -- file for .tsv sources, directory for jsons 
    bool verbose: print more info
    bool clean_docs: clean extra characters/formatting in loaded documents
    string content_field: field to use as document
    """
    
    # For content.tsv
    if data_source == 'content.tsv':

        # Initializing document preprocessor
        # TODO: update to take content_field argument
        doc_preprocessor = MemexTSVDocPreprocessor(
            path=data_loc,
            max_docs=max_docs,
            verbose=verbose,
            clean_docs=clean_docs
        )
    
    # For Elasticsearch
    elif data_source == 'es':
    
        # Initializing document preprocessor
        # TODO: update to take content_field argument
        doc_preprocessor = ESTSVDocPreprocessor(
        path=data_loc,
        max_docs=max_docs,
        verbose=verbose,
        clean_docs=clean_docs
    )

    # For MEMEX jsons -- loading from .jsonl.gz
    elif data_source == 'memex_jsons':

        # Getting all file paths
        path_list = retrieve_all_files(data_loc)

        # Applying arbitrary conditions to file path list -- here, getting .gz files
        path_list = [a for a in path_list if a.endswith('gz')]

        # Preprocessing documents from path_list
        # Set "content field" to "extracted_text" to use extracted text as raw content
        doc_preprocessor = MEMEXJsonLGZIPPreprocessor(data_loc,\
                             file_list=path_list,encoding='utf-8', max_docs=max_docs, 
                              verbose=verbose, content_field='extracted_text')
    else:
        raise ValueError('Invalid data source!')
        
    return doc_preprocessor

def create_candidate_class(extraction_type):
    """
    Creating extraction class
    
    sting extraction_type: type of extraction (options: 'location')
    """
    if extraction_type == 'location':
        # Designing candidate subclasses
        LocationExtraction = candidate_subclass('Location', ['location'])
        candidate_class = LocationExtraction
        candidate_class_name = 'LocationExtraction'
    
    return candidate_class, candidate_class_name 

def create_test_train_splits(docs, quantity, gold_dict=None, dev_frac=0.1, test_frac=0.1, seed=123, strip_end=False):
    """
    Creating train, dev, and test splits
    
    list(Document) docs: list of Document objects for partitioning
    string quantity: quantity to be extracted
    dict gold_dict: dictionary mapping urls to gold label strings (if created in preprocessing)
    float dev_frac: dev fraction
    float test_frac: test fraction
    int seed: random seed
    bool strip_end: strip end characters
    """
    # Setting set sizes
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
    if seed is not None:
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
                # If no gold_dict, use metadata for gold label quantities
                strip_end = strip_end 
                quant_ind = check_extraction_for_doc(doc, quantity, extractions_field='extractions',strip_end=strip_end)
            else:
                # Otherwise, use gold_dict
                quant_ind = doc.name in gold_list
        except:
            print('Malformatted JSON Entry!')
            quant_ind = False
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

def get_gold_labels_from_meta(session, candidate_class, target, split, annotator='gold', gold_dict=None):
    """
    Getting gold labels from
    
    Session session: DB connection
    candidate_subclass candidate_class: subclass of candidate object used to describe candidates
    string target: quantity to extract
    int split: train(0), dev(1), or test(2)
    string annotator: annotator label for snorkel DB
    gold_dict: dict mapping urls to gold label strings
    """
    
    # Getting candidates
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split).all()
    
    # Creating gold label key
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
        # Get gold label from metadata
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
        # TODO: Add other types!
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
                    
           # Originally session.query(GoldLabel).filter(GoldLabel.key == ak).filter(GoldLabel.candidate == c).first()
           # TODO: figure out how to query on GoldLabel.key without error...
            existing_label = session.query(GoldLabel).filter(GoldLabel.candidate == c).first()
            if existing_label is None:
                session.add(GoldLabel(candidate=c, key=ak, value=label))                        
                labels+=1

    pb.close()
    print("AnnotatorLabels created: %s" % (labels,))
    
    # Commit session
    session.commit()
    
def check_gold_perc(session):
    """
    Checking percentage of candidates with a positive gold label 
    (i.e. candidate extractor precision)
    
    Session session: DB connection session
    """
    gold_labels = session.query(GoldLabel).all()
    gold_vals = np.array([a.value for a in gold_labels])
    perc_pos = np.sum(gold_vals == 1)/len(gold_vals)
    print(f'Percent Positive: {perc_pos:0.2f}')
    return perc_pos

def remove_gold_labels(session):
    """
    Remove all gold labels from session
    
    Session session: DB connection
    """
    session.query(GoldLabel).delete()
