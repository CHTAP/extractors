from collections import defaultdict
import re
from fonduer.lf_helpers import get_left_ngrams, get_right_ngrams, get_between_ngrams
from snorkel.lf_helpers import get_tagged_text
import geograpy
import googlemaps as gm
from dataset_utils import city_index

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

    city = cities.cities[text.lower()]
    
    if geocode_key and city:
        gms = gm.Client(key=geocode_key)
        qo = gm.geocoding.geocode(gms, city[0])
        address = qo[0]['formatted_address']
        lat = qo[0]['geometry']['location']['lat']
        lng = qo[0]['geometry']['location']['lng']
        ext = address + ". Lat: " + str(lat) + ", Lng: " + str(lng)
    else:
        ext = city
        
    return ext

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
    
    doc_extractions = {}
    num_train_cands = max(train_marginals.shape)
    train_cand_preds = (train_marginals>0.5)*2-1
    
    cities = city_index('../utils/data/cities15000.txt')
    
    for ind in range(num_train_cands):
        if type(cands) == list:
            cand = cands[ind]
        else:
            cand = cands.get_candidate(session,ind)
        parent = cand.get_parent()
        url = parent.document.meta['url']
        doc_name = parent.document.name
   
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
                ext = getattr(cand,extraction).get_span().lower()
                if extraction == 'location':
                    ext = loc_extraction(ext, cities, geocode_key)
                if ext not in doc_extractions[doc_name][extraction]:
                    doc_extractions[doc_name][extraction].append(ext)
        if dummy:
            doc_extractions[doc_name]['dummy'].append('dummy_ext')
        
    return doc_extractions