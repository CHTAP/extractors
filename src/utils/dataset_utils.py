from snorkel.matchers import RegexMatchEach
import os
import sys
import codecs
import json
from bs4 import BeautifulSoup
import itertools
import numpy as np
import csv

import random
import numpy as np
import pandas as pd
import re
import gzip

import pycountry
import us
import editdistance

import geotext
import geograpy
from collections import defaultdict
from multiprocessing import Pool

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

def lookup_state_name(val):
    """
    Check if string is a state by direct lookup
    
    string cn: candidate text 
    """
    try:
        out = us.states.lookup(val).name
    except:
        out = 'no state'
    return out

def lookup_state_abbr(val):
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

class city_index(object):
    """
    Creates dictionary mapping possible spellings of cities to set of possible normalized cities
    City data is held as tuple: (normalized spelling, state or admin area, country, population, lat, long)
    
    Takes location of city file from geoname website
    """       
    
    cities = defaultdict(set)
    
    def __init__(self, file_loc):
        with open(file_loc, 'r') as file:

            lines = list(file)
            cities = defaultdict(set)

            for line in lines:
                columns = line.split('\t')
                value = (columns[1], columns[10], columns[8], int(columns[14]), float(columns[4]), float(columns[5]))
                
                # columns[3] is a string of comman separated alternate spellings
                for key in columns[3].lower().split(','):
                    cities[key].add(value)
                    
                    # Locations from urls are often run together
                    cities[key.replace(' ', '')].add(value)

        self.cities = cities

    def fast_loc(self, span_input):
        """
        Uses geotext and data from the geoname website to quickly detect if input is a locaiton
    
        string text: text to test for location
        """       

        span = span_input.get_span()
        reg = re.compile(r'[^a-zA-Z ]')
    
        if len(span) < 4 or reg.search(span):
            return False

        all_cities = self.cities[span.lower()]
        us_cities = [city for city in all_cities if city[2].lower() == 'us']
        city = True if us_cities else False
        
        state = span.title() in lookup_state_name(span)
        country = span.title() in lookup_country_name(span)

        return city or state or country
    
    
def set_price_data(span_input):
    """
    Sets common data used in price matchers
    
    temporary_span span_input: possible candidate to test
    """
    span = span_input.get_span().lower()
    sent = span_input.sentence.text.lower()
    right_sent = sent[span_input.char_end+1:][:15]
    left_sent = sent[:span_input.char_start][-15:]
 
    num_reg = re.compile(r'^(\d?\d[05])$')
    hour_reg = re.compile(r'\Whr\W|\Whour\W|60 min|\Wh\W')    
    half_reg = re.compile(r'half|hh|hlf|hhr|1\/2|30 min')
    quick_reg = re.compile(r'qv')
    
    return span, sent, right_sent, left_sent, num_reg, hour_reg, half_reg, quick_reg

def test_price_amount(num_reg, span, right_sent, left_sent):
    """
    Tests that candidate is a valid number that could be a price
    
    string num_reg: regex to test number
    string right_sent: sentence chunk to the right of the candidate
    string left_sent: sentence chunk to the left of the candidate
    """
    if not num_reg.search(span) or num_reg.search(span).group(0) == '00':
        return False
    
    if right_sent[:1] == '-' or left_sent[-1:] == '-':
        return False
    
    return True
    
def price_match_hour(span_input):
    """
    Uses regex to detect mentions of price
    
    string text: text to test for location
    """    
    span, sent, right_sent, left_sent, num_reg, hour_reg, half_reg, quick_reg = set_price_data(span_input)

    if not test_price_amount(num_reg, span, right_sent, left_sent):
        return False
    
    if half_reg.search(right_sent):
        return False

    if not hour_reg.search(right_sent):
        return False
    
    if half_reg.search(left_sent) and num_reg.search(right_sent):
        return False
    
    return True

def price_match_half(span_input):
    """
    Uses regex to detect mentions of price
    
    string text: text to test for location
    """    
    span, sent, right_sent, left_sent, num_reg, hour_reg, half_reg, quick_reg = set_price_data(span_input)

    if not test_price_amount(num_reg, span, right_sent, left_sent):
        return False

    if quick_reg.search(right_sent) or hour_reg.search(left_sent):
        return False
    
    if not half_reg.search(right_sent):
        return False
    
    if quick_reg.search(left_sent) and not num_reg.search(left_sent):
        return False

    return True

######################################################################################################
##### HELPER FUNCTIONS FOR PHONE	        
######################################################################################################
 # TODO: DOCSTRINGS!!!	        
def phone_cleaning (c):
    """ cleaning a candidate which has punctuations or letters, example: '8.3.2.8.9.7.8.2.1.0.&nbsp;Call' or '4143058071\\\\n'"""
    phone = re.sub("[^0-9]","", c)
    return phone

def word_to_number(span_input):	        
    num_dict = {"one":"1", 'two':"2", 'three':"3",'four':"4",'five':"5",'six':"6",'seven':"7",'eight':'8','nine':"9",'ten':'10'}

    for nb in ['one', 'two', 'three','four','five','six','seven','eight','nine','ten']:
        if span_input.find(nb):
            span_input = span_input.replace(nb,num_dict[nb] ).replace(" ","")
    return span_input

def count_(span_input, pattern):
    count = 0
    while len(span_input)>0:
        idx = span_input.find(pattern) # returns first position of character matching pattern
        span_input = span_input[idx+len(pattern):]
        if idx<0:
            break
        else:
            count+=1
    return count

def PhoneNumber( number ):
    areaCode = number[0:3 ]
    exchange = number[3:6 ]
    line = number[6:]
    return "(%s) %s-%s" % ( areaCode, exchange, line )

def arrange_phone(p):
    if len(p)==10:
        return PhoneNumber(p)
    if len(p)==11:
        return PhoneNumber(p[1:])
    if len(p) == 13:
        return PhoneNumber(p[3:])
    else:
        return p

def phone_eval(phone):
    for nb in ['one', 'two', 'three','four','five','six','seven','eight','nine','ten']:
        if count_(phone,nb)!=0:
            return arrange_phone(phone_cleaning(word_to_number(phone)))

    if phone.isdigit():
        result = arrange_phone(phone)
        return result
    else:
        phone = phone_cleaning(phone)
        if phone.isdigit():
            result = arrange_phone(phone)
            return result
        else:
            #phone =[]
            #return []
            return phone

def price_match(span_input):
        """
        Uses regex to detect mentions of price
    
        string text: text to test for location
        """       

        span = span_input.get_span().lower()

        reg = re.compile(r'^(\$?\d\d\d?[^\d,]{0,10}[^a-z0-9,]ho?u?r?|\$?\d\d\d?ho?u?r?|\$?\d\d\d?[^\d,]{0,10}60.?minutes?)$')
        match = True if reg.search(span) else False

        reg_neg = re.compile(r'half|hh|24|12|%|h\.')
        neg_match = True if reg_neg.search(span) else False

        return match and not neg_match

def fix_spacing(text):
    """
    Removes double spaces and spaces at the beginning and end of line
    
    string text: text to fix spacing on
    """    
    
    while '  ' in text:
        text = text.replace('  ',' ')
    text = text.strip()
    
    return text
    
def parse_url(text):
    """
    Finds location mentions in url and labels them.
    
    string text: string of url
    """
    
    text = clean_input(text)
    
    # Create spacing
    text = text.replace('https', '').replace('http', '').replace('com', '').replace('www', '')
    text = text.replace(':', ' ').replace('/', ' ').replace('\\', ' ').replace('.', ' ').replace('-', ' ')
    
    # This format used to mark url while encouraging Snorkel to treat the entire url as a single sentence
    url = 'Url ' + text.title() + ' <|> '
    
    url = fix_spacing(url)
    
    return url

def clean_input(text):
    """
    Removes quotes, html tags, etc
    
    string text: string to modify
    """
    
    # Strip special characters
    text = (''.join(c for c in text if ord(c) > 31 and ord(c) < 127)).encode('ascii', 'ignore').decode()
    
    # Strip html tags
    text = re.sub(r'<.*?>', ' ', text)
    # Strip html symbols
    text = re.sub('&.{,6}?;', ' ', text)
    # Strip ascii interpretation of special characters
    text = re.sub(r'\\x[a-zA-Z0-9][a-zA-Z0-9]',' ', text)

    # String literal "\n" and other such characters are in the text strings
    text = text.replace('b\'', '')
    text = text.replace('\\\'','').replace('\'','').replace('\\\"','').replace('\"','')
    text = text.replace('\\n',' ').replace('\\r',' ').replace('\\t',' ').replace('\\\\\\','')
    text = text.replace('{', ' ').replace('}', ' ').replace(';', '')
    
    text = fix_spacing(text)

    return text

# def get_posting_html_fast(text, search_term):
#     """
#     Returns ad posting from html document in memex_raw_data
    
#     string text: memex_raw_data string
#     term: regex of term to find
#     """
#     title_term = r'<[Tt]itle>(.*?)<\/[Tt]itle>'
#     body_term = r'<div.{0,20}[Pp]ost.{0,20}>(.*?)<\/div>'
#     body_term2 = r'<p>(.*?)<\/p>'

#     title = re.search(title_term, text)
#     html_lines = [clean_input(line) for line in (re.findall(body_term, text) + re.findall(body_term2, text))]
# #    search_lines = [clean_input(line) for line in re.findall(search_term, text)]
    
#     if title and title.group(1):
#         title = clean_input(title.group(1))
#     else:
#         title = '-1'
        
#     html_text = 'Title ' + title.replace('.', ' ').replace(':', ' ') + ' <|> '
#     for line in html_lines:
#         if line:
#             html_text += ' ' + line + ' <|> '
# #    for line in search_lines:
# #        if line:
# #            html_text += ' Search' + line.replace('.', ' ').replace(':', ' ') + ' <|> '
        
#     html_text = fix_spacing(html_text)

#     return html_text

def get_posting_html_fast(text, search_term):
    """
    Returns ad posting from html document in memex_raw_data
    
    string text: memex_raw_data string
    term: regex of term to find
    """
    title_term = r'<[Tt]itle>([\w\W]*?)</[Tt]itle>'
    body_term = r'<div.{0,20}[Cc]ontent.{0,20}>([\w\W]*?)</div>'
    body_term2 = r'<div.{0,20}[Pp]ost.{0,20}>([\w\W]*?)</div>'
    body_term3 = r'<div.{0,20}[Tt]ext.{0,20}>([\w\W]*?)</div>'
    body_term4 = r'<p>([\w\W]*?)</p>'

    title = re.search(title_term, text)
    body_lines = re.findall(body_term, text) + re.findall(body_term2, text) + re.findall(body_term3, text) + re.findall(body_term4, text)
    html_lines = [clean_input(line) for line in body_lines]
    
    if title and title.group(1):
        title = clean_input(title.group(1))
    else:
        title = '-1'

    html_text = 'Title ' + title.replace('.', ' ').replace(':', ' ') + ' <|> '

    for line in html_lines:
        if line:
            html_text += ' ' + line + ' <|> '

    if search_term:
        search_lines = [clean_input(line) for line in re.findall(search_term, text)]
        for line in search_lines:
            if line:
                html_text += ' Search ' + line.replace('.', ' ').replace(':', ' ') + ' <|> '
        
    html_text = fix_spacing(html_text)
    
    return html_text

def get_posting_html(text, term):
    """
    Returns ad posting from html document in memex_raw_data
    
    string text: memex_raw_data string
    term: regex of term to find
    """

    term_reg = re.compile(term)
    html = BeautifulSoup(text, 'lxml')
    title = html.find('title')
    html_lines = list(html.findAll(text=True))
    html_text = ""
    
    if title and title.string:
        clean_title = clean_input(title.string)
    else:
        clean_title = ''

    # These values were deterimined experimentally
    line_min = 30
    spec_max = .1
    increment = .2

    for line in html_lines:
        clean_line = clean_input(line)
        has_term = term_reg.search(clean_line.lower())

        # Lines longer than a certain length are more likely to be the substanitive part of the post
        # term_reg gives the option to override line length if a key term is contained
        if len(clean_line) >= line_min or has_term:
            
            # Creates a line with only special characters
            spec_line = re.sub(r'[\w ]*', '', clean_line)

            # Lines with a high proportion of special characters are much more likely to be non-text
            spec_clean_ratio = len(spec_line)/max(1, len(clean_line))
        
            # {} are much more likely to be in a non-text line
            if '{' in spec_line or '}' in spec_line:
                spec_clean_ratio += increment

            # OBJ_S marks the start of the term, OBJ_E marks the end
            if spec_clean_ratio < spec_max and clean_title and clean_line == clean_title:
                html_text += ' Title ' + clean_line.replace('.', ' ').replace(':', ' ') + ' <|> '
            elif spec_clean_ratio < spec_max and has_term:
                html_text += ' Search ' + clean_line.replace('.', ' ').replace(':', ' ') + ' <|> '
            elif spec_clean_ratio < spec_max:
                html_text += ' ' + clean_line + ' <|> '

    html_text = fix_spacing(html_text)
    
    return html_text

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

    # Run twice for cases of regex overlap
    for iteration in range(2):
        #Strips double quotes that don't start or end strings that will make up keys or values
        indices = [m.start(0) for m in re.finditer(r'[^:,][^{]"[^:,}]', text)]
        for ii in indices:
            #+2 because the regex above will catch two characters before the double quote
            text = replace_str_index(text,ii+2,"'")    
        
    return text

def clean_extracted_text(text):
    """
    Cleans text extracted from html files -- currently works with data from MEMEX crawl
    
    string text: string to modify
    """
    
    # Replacing and stripping special characters
    text = text.replace('\\n','').replace('\'','"').replace('|','').strip('\n').strip('\r').strip('b').strip('"').replace('\\\"','"').replace('""','"')

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
    if extraction_dict != -1 and quantity in list(extraction_dict.keys()):
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
                    
class ParallelESTSVPreprocessor(HTMLDocPreprocessor):
    
    def __init__(self, path, encoding="utf-8", max_docs=float('inf'), verbose=False, clean_docs=False,
                 content_fields=['extracted_text'], term='', max_doc_length=0, data_source='es'):
        #self.encoding = encoding
        #self.max_docs = max_docs
        self.path = path
        self.clean_docs = clean_docs
        self.verbose = verbose
        self.content_fields=content_fields
        self.term=term
        self.max_doc_length=max_doc_length
        self.data_source=data_source
        super().__init__(path, encoding=encoding, max_docs=max_docs)
        
    def _get_files(self,path_list):
        fpaths = [fl for fl in path_list]
        return fpaths
    
    def _can_read(self, fpath):
        return fpath.endswith('tsv') or fpath.endswith('csv')
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        file_list = os.listdir(self.path)
        file_list = [os.path.join(self.path, fl) for fl in file_list]
        for file_name in self._get_files(file_list):
            if self._can_read(file_name):
                for doc, text in self.parse_file(file_name):
                    yield doc, text
                    doc_count += 1
                    if self.verbose:
                        print(f'Parsed {doc_count} docs...')
                    if doc_count >= self.max_docs:
                        return

    def parse_file(self, fp):
        i=0
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for ind, line in enumerate(tsv):
                if ind == 0:
                    fields = line.split('\t')
                    num_fields = len(fields)
                try:
                    # Loading data -- ignore malformatted entries!
                    # TODO: Make these fields dynamic/drawn from header? Or make field names an option?
                   # if num_fields == 21:
                   #     (doc_id, uuid, memex_id, memex_content_type, crawl_data, memex_crawler, memex_doc_type, memex_extracted_metadata, memex_extracted_text, memex_extractions, memex_raw_content, memex_team, memex_timestamp, memex_type, memex_url, memex_version, domain, content_type, url, content, extractions) = line.split('\t')
                    #elif num_fields == 8:
                    if self.data_source='spark':
                        (doc_id, memex_doc_type, memex_url, memex_url_parsed, memex_raw_content, extracted_phone, extracted_age, extracted_rate, extracted_ethnicity, extracted_email, extracted_incall) = line.split('\t')
                    elif self.data_source='es':
                        (doc_id, uuid, memex_id, memex_doc_type, memex_raw_content, memex_url, url, extractions) = line.split('\t')
                    content = None
                except:
                    print('Malformatted Line!')
                    continue
                
                # Cleaning documents if specified
                if self.clean_docs:
                    if 'extracted_text' in self.content_fields:
                        content = clean_input(content)
                    if 'raw_content' in self.content_fields:
                        memex_raw_content = get_posting_html_fast(memex_raw_content, self.term)
                if 'url' in self.content_fields:
                    memex_url = parse_url(memex_url)

                doc_text = ""
                field_names = {'extracted_text': content, 'raw_content': memex_raw_content, 'url': memex_url, }
                for field in self.content_fields:
                    doc_text += field_names[field] + " "

                doc_text = doc_text.strip()
                doc_name = doc_id
                extractions = extractions
                
                # Short documents are usually parsing errors...
                if len(doc_text) < 10:
                    if self.verbose:
                        print('Short Doc!')
                    continue
       
                # long documents sometimes causing parsing to stall
                if self.max_doc_length and len(doc_text) > self.max_doc_length:
                    if self.verbose:
                        print('Long document')
                    continue
                    #doc_text = doc_text[-self.max_doc_length:]
                
                # Setting stable id
                stable_id = self.get_stable_id(doc_name)
            
                # Yielding results, adding useful info to metadata
                doc = Document(
                    name=doc_name, stable_id=stable_id,
                    meta={'extractions':extractions,
                          'url':url}
                )
                yield doc, doc_text                    

class ESTSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file drawn from Elasticsearch"""
    
    def __init__(self, path, encoding="utf-8", max_docs=float('inf'), verbose=False, clean_docs=False,
                 content_fields=['extracted_text'], term='', max_doc_length=0, data_source=data_source):
        super().__init__(path, encoding=encoding, max_docs=max_docs)
        self.verbose = verbose
        self.clean_docs = clean_docs
        self.content_fields=content_fields
        self.term=term
        self.max_doc_length=max_doc_length
        self.data_source=data_source  
    def _get_files(self,path_list):
        fpaths = [fl for fl in path_list]
        return fpaths
    
    def _can_read(self, fpath):
        return fpath.endswith('tsv') or fpath.endswith('csv')
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        file_list = os.listdir(self.path)
        file_list = [os.path.join(self.path, fl) for fl in file_list]
        for file_name in self._get_files(file_list):
            if self._can_read(file_name):
                for doc, text in self.parse_file(file_name):
                    yield doc, text
                    doc_count += 1
                    if self.verbose and (doc_count % 1000 == 0):
                        print(f'Parsed {doc_count} docs...')
                    if doc_count >= self.max_docs:
                        return

    def parse_file(self, fp):
        i=0
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for ind, line in enumerate(tsv):
                if ind == 0:
                    num_fields = len(line.split('\t')) 
                    continue
                try:
                    # Loading data -- ignore malformatted entries!
                    # TODO: Make these fields dynamic/drawn from header? Or make field names an option?
                    if self.data_source='spark':
                        (doc_id, memex_doc_type, memex_url, memex_url_parsed, memex_raw_content, extracted_phone, extracted_age, extracted_rate, extracted_ethnicity, extracted_email, extracted_incall) = line.split('\t')
                    elif self.data_source='es':
                        (doc_id, uuid, memex_id, memex_content_type, crawl_data, memex_crawler, memex_doc_type, memex_extracted_metadata, memex_extracted_text, memex_extractions, memex_raw_content, memex_team, memex_timestamp, memex_type, memex_url, memex_version, domain, content_type, url, content, extractions) = line.split('\t')
                    content = None
                except:
                    print('Malformatted Line!')
                    continue

                # Cleaning documents if specified
                if self.clean_docs:
                    if 'extracted_text' in self.content_fields:
                        content = clean_input(content)
                    if 'raw_content' in self.content_fields:
                        memex_raw_content = get_posting_html_fast(memex_raw_content, self.term)
                if 'url' in self.content_fields:
                    memex_url = parse_url(memex_url)

                doc_text = ""
                field_names = {'extracted_text': content, 'raw_content': memex_raw_content, 'url': memex_url, }
                for field in self.content_fields:
                    doc_text += field_names[field] + " "

                doc_text = doc_text.strip()
                doc_name = doc_id
                source = content_type
                extractions = extractions
                
                # Short documents are usually parsing errors...
                if len(doc_text) < 10:
                    if self.verbose:
                        print('Short Doc!')
                    continue
       
                # long documents sometimes causing parsing to stall
                if self.max_doc_length and len(doc_text) > self.max_doc_length:
                    if self.verbose:
                        print('Long document')
                    # continue
                    doc_text = doc_text[-self.max_doc_length:]
 
                # Setting stable id
                stable_id = self.get_stable_id(doc_name)
                
                # Yielding results, adding useful info to metadata
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
                    if self.verbose and (doc_count % 1000 == 0):
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

def parallel_parse_html(path, term='', threads=30, col=10):
    """
    Creates new tsv file with html field replaced with a parsed version from get_posting_html
    
    string path: path to data source -- folder of .tsv files
    string term: regex supplied to get_posting_html
    int col: column of html to parse
    int threads: number of threads to create
    """
    pool = Pool(threads) 

    file_list = os.listdir(path)
    path_list = [os.path.join(path, file) for file in file_list]
    file_data = [(path, term, col) for path in path_list if path.endswith('tsv')]

    out_dir = path + '/parsed/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = pool.map(parse_html, file_data)

    pool.close()
    pool.join()

def parse_html(file_data):
    """
    Creates new tsv file with html field replaced with a parsed version from get_posting_html
    
    tuple file_data: (path to data source -- .tsv file, regex supplied to get_posting_html, column of html to parse)
    """
    in_loc, term, col = file_data
    csv.field_size_limit(sys.maxsize)
    
    out_dir = os.path.dirname(in_loc) + '/parsed/'
    out_loc = out_dir + os.path.basename(in_loc)
    
    with open(in_loc, 'r') as in_file, open(out_loc, 'w') as out_file:
        reader = csv.reader(in_file, delimiter='\t')
        writer = csv.writer(out_file, delimiter='\t')
        
        first_line = next(reader)
        writer.writerow(first_line)
        for i, line in enumerate(reader):
            try:
                line[col] = get_posting_html_fast(line[col], term)
                writer.writerow(line)
            except:
                print('Error on line: ' + str(i))

def combine_dedupe(dev_loc, added_train_docs, out_loc):
    """
    Creates new tsv file by combining dev_loc and added_train_docs without duplicates.
    
    string dev_loc: path to data source -- .tsv file
    string added_train_docs: path to data source -- .tsv file
    string out_loc: path to output file -- .tsv file
    """
    
    csv.field_size_limit(sys.maxsize)

    with open(dev_loc, 'r') as dev_file, open(added_train_docs, 'r') as train_file, open(out_loc, 'w') as outfile:
        dev_reader = csv.reader(dev_file, delimiter='\t')
        devs = list(dev_reader)
        train_reader = csv.reader(train_file, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        for line in devs:
            writer.writerow(line)

        for train_line in train_reader:
            for dev_line in devs:
                if line[0] != dev_line[0]:
                    writer.writerow(line)
    
    return out_loc

def set_preprocessor(data_source,data_loc,max_docs=1000,verbose=False,clean_docs=True,content_fields=['extracted_text'],
                     term='', max_doc_length=0): 
    """
    Sets a chosen document preprocessor.
    
    string data_source: type of data soure -- 'content.tsv', 'es' (Elasticsearch), 'memex_jsons'
    string data_loc: path to data source -- file for .tsv sources, directory for jsons 
    bool verbose: print more info
    bool clean_docs: clean extra characters/formatting in loaded documents
    list of strings content_field: fields to use as document
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
    elif data_source == 'es' or 'spark':
    
        # Initializing document preprocessor
        
        if '.tsv' in data_loc or '.csv' in data_loc:
            print('Using single-threaded loader')
            doc_preprocessor = ESTSVDocPreprocessor(
            path=data_loc,
            max_docs=max_docs,
            verbose=verbose,
            clean_docs=clean_docs,
            content_fields=content_fields,
            term=term,
            max_doc_length=max_doc_length,
            data_source=data_source
        )
        else:
            print('Using parallelized loader')
            doc_preprocessor = ParallelESTSVPreprocessor(
            path=data_loc,
            max_docs=max_docs,
            verbose=verbose,
            clean_docs=clean_docs,
            content_fields=content_fields,
            term=term,
            max_doc_length=max_doc_length,
            data_source=data_source
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

    if extraction_type == 'phone':
        PhoneExtraction = candidate_subclass('Phone', ['phone'])	
        candidate_class = PhoneExtraction
        candidate_class_name = 'PhoneExtraction'

    if extraction_type == 'price':
        # Designing candidate subclasses
        PriceExtraction = candidate_subclass('Price', ['price'])
        candidate_class = PriceExtraction
        candidate_class_name = 'PriceExtraction'
        
    if extraction_type == 'email':
        # Designing candidate subclasses
        PriceExtraction = candidate_subclass('Email', ['email'])
        candidate_class = PriceExtraction
        candidate_class_name = 'EmailExtraction'
        
    if extraction_type == 'age':
        # Designing candidate subclasses
        AgeExtraction = candidate_subclass('Age', ['age'])
        candidate_class = AgeExtraction
        candidate_class_name = 'AgeExtraction'
        
    if extraction_type == 'call':
        # Designing candidate subclasses
        CallExtraction = candidate_subclass('Call', ['call'])
        candidate_class = CallExtraction
        candidate_class_name = 'CallExtraction'
        
    if extraction_type == 'ethnicity':
        # Designing candidate subclasses
        EthnicityExtraction = candidate_subclass('Ethnicity', ['ethnicity'])
        candidate_class = EthnicityExtraction
        candidate_class_name = 'EthnicityExtraction'
    
    return candidate_class, candidate_class_name 

def create_test_train_splits(docs, quantity, gold_dict=None, dev_frac=0.1, test_frac=0.1, seed=123, strip_end=False, hand_label=False):
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
            if hand_label:
                quant_ind = True
            elif gold_dict is None:
                # If no gold_dict, use metadata for gold label quantities
                strip_end = strip_end 
                quant_ind = check_extraction_for_doc(doc, quantity, extractions_field='extractions',strip_end=strip_end)
            else:
                # Otherwise, use gold_dict
                quant_ind = doc.name in gold_list
        except:
            #print('Malformatted JSON Entry!')
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

        elif target == 'phone':
            val = phone_eval(ext.get_span())
            clean_value = re.sub('[^A-Za-z0-9]+', '', val)
            clean_gold_value = re.sub('[^A-Za-z0-9]+', '', target_strings)
            if (clean_value in clean_gold_value) or (clean_gold_value in clean_value):
                label = 1
            else:
                label = -1

        else:
            raise ValueError('Unrecognized extraction type!')

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
