from collections import defaultdict
import re
from fonduer.lf_helpers import get_left_ngrams, get_right_ngrams, get_between_ngrams
from snorkel.lf_helpers import get_tagged_text
import geograpy
import googlemaps as gm


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
def loc_extraction(text, geocode_key=None):
    """
    If text is a city, returns formatted address and geocode, else returns text

    string text: location to format
    string geocode_key: Googlemaps api key
    """

    city = geograpy.get_place_context(text=text.title()).cities
    
    if geocode_key and city:
        gms = gm.Client(key=geocode_key)
        qo = gm.geocoding.geocode(gms, city[0])
        address = qo[0]['formatted_address']
        lat = qo[0]['geometry']['location']['lat']
        lng = qo[0]['geometry']['location']['lng']
        ext = address + ". Lat: " + str(lat) + ", Lng: " + str(lng)
    else:
        ext = text.title()
        
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
                    ext = loc_extraction(ext, geocode_key)
                if ext not in doc_extractions[doc_name][extraction]:
                    doc_extractions[doc_name][extraction].append(ext)
        if dummy:
            doc_extractions[doc_name]['dummy'].append('dummy_ext')
        
    return doc_extractions
#####################################################################
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
######################################################################################################
##### HELPER FUNCTIONS FOR EXTRACTIONS
######################################################################################################

def create_phone_extractions_dict(session, cands, train_marginals, extractions, dummy=False):
    """
    Creating dictionary of extractions from label matrix and marginals.
    
    session: DB connection
    LabelMatrix or List cands: training label matrix
    list(int) train_marginals: marginal probablities emitted from Snorkel
    list extractions: list of extractions to add
    bool dummy: include dummy extraction
    """
    
    doc_extractions = {}
    num_train_cands = max(train_marginals.shape)
    train_cand_preds = (train_marginals>0.5)*2-1
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
                ext = phone_eval(getattr(cand,extraction).get_span().lower())
                #clean_ext = ext(phone_eval(ext))
                doc_extractions[doc_name][extraction].append(ext)
        if dummy:
            doc_extractions[doc_name]['dummy'].append('dummy_ext')
        
    return doc_extractions
