from collections import defaultdict
import re
import numpy as np
from snorkel.lf_helpers import get_tagged_text
import geograpy
import googlemaps as gm
from dataset_utils import city_index, phone_eval
from operator import itemgetter
from nltk.corpus import words
from multiprocessing import Pool
from functools import partial
from itertools import chain

######################################################################################################
##### HELPER FUNCTIONS FOR LABELING FUNCTIONS
######################################################################################################

def ltp(x):
    """
    Transforming list to parenthetical
    """
    return '(' + '|'.join(x) + ')'

def rule_regex_search_before_A(candidate, pattern, sign):
    """
    Check if regex before expresision A
    """
    return sign if re.search(pattern + r'*{{A}}', get_tagged_text(candidate), flags=re.I) else 0

def overlap(a, b):
    """Check if a overlaps b.
    This is typically used to check if ANY of a list of phrases is in the ngrams returned by an lf_helper.
    :param a: A collection of items
    :param b: A collection of items
    :rtype: boolean
    """
    return not set(a).isdisjoint(b)

######################################################################################################
##### HELPER FUNCTIONS FOR EXTRACTIONS
######################################################################################################
def loc_extraction(text, cities, geocode_key=None):
    """
    If text is a city, returns formatted address and geocode, else returns text

    string text: location to format
    string geocode_key: Googlemaps api key
    """

    cities = list(cities.cities[text.lower()])
    if cities:
        city = sorted(cities, key=itemgetter(3))[-1]
    else:
        city = ''
    
    if geocode_key and city:
        gms = gm.Client(key=geocode_key)
        qo = gm.geocoding.geocode(gms, city[0])
        address = qo[0]['formatted_address']
        lat = qo[0]['geometry']['location']['lat']
        lng = qo[0]['geometry']['location']['lng']
        ext = (address, lat, lng)
    else:
        ext = city
        
    ext = [f'Extracted String: {text}']+list(ext)
    return ext

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield np.arange(i, i + n, 1)
        
def create_extractions_dict_parallel(session, cands, train_marginals, extractions, dummy=False, geocode_key=None, slices=1):
    """
    Parallelizing creating extractions dictionary
    """
    if slices == 1:
        print("Using serial processing for extractions...")
        doc_extractions = create_extractions_dict(session, cands, train_marginals, extractions, dummy=False, geocode_key=None)
    else:
        print("Using parallel processing for extractions...")
        pool = Pool(slices)
        sz = int(len(train_marginals)/slices)
        marg_chunks = []
        cand_chunks = []
        for c in chunks(train_marginals, sz):
            c = [a for a in c if a<len(train_marginals)]
            cand_chunks.append([cands[a] for a in c])
            marg_chunks.append([train_marginals[a] for a in c])
        func = partial(create_extractions_dict, session=session, extractions=extractions, dummy=False, geocode_key=geocode_key)
        doc_extractions_parallel = pool.map(func, cands=cand_chunks, train_marginals=marg_chunks)
        doc_extractions = {}
        for e in doc_extractions_parallel:
            doc_extractions.update(e)
        
    return doc_extractions
        
    
    
def create_extractions_dict(session, cands, train_marginals, extractions, dummy=False, geocode_key=None):
    """
    Creating dictionary of extractions from label matrix and marginals.
    
    session: DB connection
    LabelMatrix or List cands: training label matrix
    list(int) train_marginals: marginal probablities emitted from Snorkel
    list extractions: list of extractions to add
    bool dummy: include dummy extraction
    string geocode_key: Googlemaps api key
    """
    print("Starting extractions dictionary creation...")
    doc_extractions = {}
    num_train_cands = max(train_marginals.shape)
    train_cand_preds = (train_marginals>0.5)*2-1
    
    cities = city_index('../utils/data/cities15000.txt')
    
    for ind in range(num_train_cands):
        if ind % 1000 == 0:
            print(f"Processing extraction {ind} of {num_train_cands}...")
        if type(cands) == list:
            cand = cands[ind]
        else:
            cand = cands.get_candidate(session,ind)
        url = cand.document.meta['url']
        doc_name = cand.document.name

        # Initializing key if it doesn't exist
        if doc_name not in doc_extractions.keys():
            doc_extractions[doc_name] = {}
            doc_extractions[doc_name]['url'] = url
            for extraction in extractions:
                doc_extractions[doc_name][extraction] = []
            if dummy:
                doc_extractions[doc_name]['dummy'] = []
        
        # Adding extraction to extractions dict if prediction == 1
        if train_cand_preds[ind] == 1:
            for extraction in extractions:
                mention_type = f"{extraction}_mention"
                ext = getattr(cand,mention_type).context.get_span().lower()
                if extraction == 'phone':
                    ext = phone_eval(ext)
                    doc_extractions.append[doc_name][extraction].append(ext)
                if extraction == 'location':
                    geocode = loc_extraction(ext, cities, geocode_key)
                    confidence = 1 if ext in url else 0
                    #if ext in words.words():
                         #confidence -= 1
                    if not doc_extractions[doc_name].get(extraction):
                        doc_extractions[doc_name][extraction] = (geocode, confidence)
                    elif doc_extractions[doc_name][extraction][1] < confidence:
                        doc_extractions[doc_name][extraction] = (geocode, confidence)
                elif extraction == 'price':
                    reg_cost = re.compile(r'\d\d\d?')
                    cost = reg_cost.search(ext).group(0) + '/hour'
                    if not doc_extractions[doc_name][extraction]:
                        doc_extractions[doc_name][extraction] = cost
                elif ext not in doc_extractions[doc_name][extraction]:
                    doc_extractions[doc_name][extraction].append(ext)
        
        if dummy:
            doc_extractions[doc_name]['dummy'].append('dummy_ext')
        
    return doc_extractions
