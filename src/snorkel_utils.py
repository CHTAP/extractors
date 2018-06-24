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
            

######################################## MATCHERS ###################################
#Importing matchers module and defining LocationMatchers
import re
from snorkel.lf_helpers import *
from snorkel.matchers import RegexMatchEach, Union, LambdaFunctionMatcher
import pdb
import us
from geotext import GeoText
from builtins import range
import csv
import codecs
import pycountry
import us
import editdistance
from nltk.corpus import stopwords

# HACK. GET RID OF THESE
from fonduer.lf_helpers import *



# these are sths to think about for future 
stop_words = [w for w in set(stopwords.words('english')) if len(w)<=3]
es_stop_words = [w for w in set(stopwords.words('spanish'))]
for w in ["100","pm","ad","id","tv", "hr","address","videos","premium","adress","hi",\
         "de","como","toes","most","bar","un","una","de","mx","la","of","gentelmane","come",\
          "spa","com","man","ms","mr","br","most","oral","money","hr","much","game","kg","lb","ar","min","max","men",\
          "ok","si","cam","roses","height","weight","hon","baby","honey","asian","ukrainian", "hispanic","black",\
          "white","ethnicity","came","incallz","outcall","incall"]:
    stop_words.append(w)
for es in es_stop_words:
    stop_words.append(es)

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


# def cardinal_direction_matcher(span_input):
#     cardinal_loc= ["north","west","east","south","northeast","NE",\
#                    "southeast","SE", "southwest","SW","northwest","NW"]
#     span_input = span_input.get_span()
#     splitted_span= span_input.split()
#     for s in splitted_span:
#         if s in cardinal_loc:
                 
#             return True    
#         else:
#             return False

#cardinal_direction_lambda_matcher =LambdaFunctionMatcher(func=cardinal_direction_matcher)

def lookup_city_matcher(span_input):
    span = span_input.get_span()
    if span.lower() in stop_words:
        return False
    
    span_input = span.upper()
    places = GeoText(span_input)
    lst = places.cities
    
    if len(lst)!=0:
        if lst[0] in ['BUY', 'DATE','LATINA','MOBILE',"ANNA","DEAL","MALE","MESA","BEST","UVA","YOUNG","SALE","SPLIT","BOO"\
                     ,"NICE","SAME","OK","COME","SETCOOKIE","SPA","BR","ORAL","HI","GENTELMANE","DE","COMO","OF",\
                      "TOES","MOST","BAR","OK"]:
            return False
        return True    
    else:
        return False

def lookup_country_name_matcher(span_input):
    span = span_input.get_span()
    if span.lower() in stop_words:
        return False
    
    if span.upper() in ['BUY', 'DATE','LATINA','MOBILE',"ANNA","DEAL","MALE","MESA","BEST","UVA","YOUNG","SALE","SPLIT","BOO"\
                     ,"NICE","SAME","OK","COME","SETCOOKIE","SPA","BR","ORAL","HI","GENTELMANE","DE","COMO","OF",\
                      "TOES","MOST","BAR","OK"]:
        return False
    try:
        out = pycountry.countries.lookup(span).name
        return True
    except:    
        return False
    
def lookup_state_name_matcher(span_input):
    span = span_input.get_span()
    if span.lower() in stop_words:
        return False
    
    try:
        out = us.states.lookup(span).name
        return True
        
    except:        
        return False

########## THROTTLERS ########

# Creating filter to eliminate mentions of currency  
def location_currencies_filter(location):
    list_currencies = [ "dollar", "dollars", "lira","kwacha","rials","rial","dong","dongs","fuerte","euro",
                       "euros","vatu","som","peso","sterling","sterlings","soms","pestos","ok",
                       "pounds", 
                  "pound","dirham","dirhams","hryvnia","manat","manats","liras","lira",
                       "dinar","dinars","pa'anga","franc","baht","schilling",
                  "somoni","krona","lilangeni","rupee","rand","shilling","leone","riyal","dobra",
                  "tala","ruble","zloty","peso","sol","quarani","kina","guinean","balboa","krone","naira",
                  "cordoba","kyat","metical","togrog","leu","ouguiya","rufiyaa","ringgit","kwacha",
                  "ariary","denar","litas","loti","lats","kip","som","won","tenge","yen","shekel","rupiah",
                  "forint","lempira","gourde","quetzal","cedi","lari","dalasi","cfp","birr","kroon","nakfa",
                  "cfa","Peso","koruna","croatian","colon","yuan","escudo","cape","riel","lev","real"
                  ,"real","mark","boliviano","ngultrum","taka","manat","dram","kwanza","lek","afghani","renminbi"]

    
    cand_right_tokens = list(get_right_ngrams(location,window=2))
    for cand in cand_right_tokens:
        if cand not in list_currencies:
            return True
        
def filter_capital_words(c):
    patern = re.compile("(?=.{1,20}$)[A-Z](\s*?[A-Z])*$")
    span = c[0].get_span()
    result = patern.match(span)
    if result:
        return False

def filter_characters(c):
    patern_1 = re.compile("^[(+*\?/\-,@/$)]$")
    patern_2 = re.compile ("^[(+*?/\-)]\s*[(A-Z)]*[(a-z)]*$")
    patern_3 = re.compile("@[a-z]*|[1-9]")
    patern_4 = re.compile("(?=.{1,20})[A-Z]*[(+*\?/\-&)]")
    patern_5 = re.compile("([a-z]|[A-Z])*[((+*\?/\-,@/$)]*([1-9])\d{3}")
    patern_6 = re.compile("([A-Z]|[a-z_])*(\?|\$|\.|\-)[^\s]*$")
    
    patern_7 = re.compile(".+\_.+\_+.+\_+.")
    span = c[0].get_span()
    result_1 = patern_1.match(span)
    result_2 = patern_2.match(span)
    result_3 = patern_3.match(span)
    result_4 = patern_4.match(span)
    result_5 = patern_5.match(span)
    result_6 = patern_6.match(span)
    result_7 = patern_7.match(span)
    if result_1 or result_2 or result_3 or result_4 or result_5 or  result_6 or result_7:
        return False

def filter_numbers(c):
    patern_1 = re.compile("^[1-9][(\')][1-9][(\")]$")
    patern_2 = re.compile("^[1-9]*$")
    span = c[0].get_span()
    result_1 = patern_1.match(span)
    result_2 = patern_2.match(span)
    if result_1 or result_2:
        return False


def filter_emaile_website(c):
    patern_1 = re.compile(".+\@.+\..+")
    patern_2 = re.compile(".+\..+\..+")
    span = c[0].get_span()
    result_1 = patern_1.match(span)
    result_2 = patern_2.match(span)
    if result_1 or result_2:
        return False

def filter_(c):
    if location_currencies_filter(c) == False:
        return False
    if filter_capital_words(c) == False:
        return False
    if filter_characters(c) == False:
        return False
    if filter_numbers(c) == False:
        return False
    if filter_emaile_website(c) == False:
        return False
    return True

def get_candidate_filter():
    return filter_

##### GETTING OVERALL CANDIDATE EXTRACTOR ######

def get_location_matcher():

    location_matcher_1 = LocationMatcher(longest_match_only=True)     
    city_lambda_matcher = LambdaFunctionMatcher(func=lookup_city_matcher)
    country_lambda_matcher =LambdaFunctionMatcher(func=lookup_country_name_matcher)
    state_lambda_matcher =LambdaFunctionMatcher(func=lookup_state_name_matcher)
    location_matcher_ = Union(country_lambda_matcher,state_lambda_matcher,location_matcher_1,city_lambda_matcher) 
    return location_matcher_


#### PULLING CANDIDATE EXTRACTOR CLASS FROM FONDUER WITH FILTER ######

from snorkel.udf import UDF, UDFRunner
from copy import deepcopy
from itertools import product

class CandidateExtractorUDF(UDF):
    def __init__(self, candidate_class, cspaces, matchers, candidate_filter, self_relations, nested_relations, symmetric_relations, **kwargs):
        self.candidate_class     = candidate_class
        # Note: isinstance is the way to check types -- not type(x) in [...]!
        self.candidate_spaces    = cspaces if isinstance(cspaces, (list, tuple)) else [cspaces]
        self.matchers            = matchers if isinstance(matchers, (list, tuple)) else [matchers]
        self.candidate_filter    = candidate_filter
        self.nested_relations    = nested_relations
        self.self_relations      = self_relations
        self.symmetric_relations = symmetric_relations

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Make sure the candidate spaces are different so generators aren't expended!
        self.candidate_spaces = list(map(deepcopy, self.candidate_spaces))

        # Preallocates internal data structures
        self.child_context_sets = [None] * self.arity
        for i in range(self.arity):
            self.child_context_sets[i] = set()

        super(CandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, **kwargs):
        # Generate TemporaryContexts that are children of the context using the candidate_space and filtered
        # by the Matcher
        for i in range(self.arity):
            self.child_context_sets[i].clear()
            for tc in self.matchers[i].apply(self.candidate_spaces[i].apply(context)):
                tc.load_id_or_insert(self.session)
                self.child_context_sets[i].add(tc)

        # Generates and persists candidates
        extracted = set()
        candidate_args = {'split': split}
        for args in product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]):

            # Apply candidate_filter if one was given
            # Accepts a tuple of Context objects (e.g., (Span, Span))
            # (candidate_filter returns whether or not proposed candidate passes throttling condition)
            if self.candidate_filter:
                if not self.candidate_filter(tuple(args[i][1] for i in range(self.arity))):
                    continue

            # TODO: Make this work for higher-order relations
            if self.arity == 2:
                ai, a = args[0]
                bi, b = args[1]

                # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                # "symmetric" relations. For symmetric relations, if mentions are of the same type, maintain
                # their order in the sentence.
                if not self.self_relations and a == b:
                    continue
                elif not self.nested_relations and (a in b or b in a):
                    continue
                elif not self.symmetric_relations and ((b, a) in extracted or
                    (self.matchers[0] == self.matchers[1] and a.char_start > b.char_start)):
                    continue

                # Keep track of extracted
                extracted.add((a,b))

            # Assemble candidate arguments
            for i, arg_name in enumerate(self.candidate_class.__argnames__):
                candidate_args[arg_name + '_id'] = args[i][1].id

            # Checking for existence
            if not clear:
                q = select([self.candidate_class.id])
                for key, value in iteritems(candidate_args):
                    q = q.where(getattr(self.candidate_class, key) == value)
                candidate_id = self.session.execute(q).first()
                if candidate_id is not None:
                    continue

            # Add Candidate to session
            yield self.candidate_class(**candidate_args)

class CandidateExtractorFilter(UDFRunner):
    """
    An operator to extract Candidate objects from a Context.
    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param candidate_filter: an optional function for filtering out candidates which returns a Boolean expressing whether or not
                      the candidate should be instantiated.
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is False.
    """
    def __init__(self, candidate_class, cspaces, matchers, candidate_filter=None, self_relations=False, nested_relations=False, symmetric_relations=False):
        super(CandidateExtractorFilter, self).__init__(CandidateExtractorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 candidate_filter=candidate_filter,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        super(CandidateExtractorFilter, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        session.query(Candidate).filter(Candidate.split == split).delete()

##################### FOR RNN ###################
class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1): 
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in iteritems(self.d)}
    
from snorkel.learning.disc_models.rnn import TextRNN
class MemexTextRNN(TextRNN):
    """TextRNN for strings of text."""
    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences
        
        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
        data, ends = [], []
        for candidate in candidates:
            # Small change: used get_span() instead of text below
            toks = candidate.get_contexts()[0].get_span().split()
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(list(map(f, toks))))
            ends.append(len(toks))
        return data, ends
