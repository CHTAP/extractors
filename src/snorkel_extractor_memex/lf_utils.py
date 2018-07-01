from collections import defaultdict
import re
from fonduer.lf_helpers import get_left_ngrams, get_right_ngrams, get_between_ngrams
from snorkel.lf_helpers import get_tagged_text


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

def create_extractions_dict(session, L_train, train_marginals, extractions, dummy=False):
    """
    Creating dictionary of extractions from label matrix and marginals.
    
    session: DB connection
    LabelMatrix L_train: training label matrix
    list(int) train_marginals: marginal probablities emitted from Snorkel
    list extractions: list of extractions to add
    bool dummy: include dummy extraction
    """
    
    doc_extractions = {}
    num_train_cands = max(train_marginals.shape)
    train_cand_preds = (train_marginals>0.5)*2-1
    for ind in range(num_train_cands):
        cand = L_train.get_candidate(session,ind)
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
                doc_extractions[doc_name][extraction].append(ext)
        if dummy:
            doc_extractions[doc_name]['dummy'].append('dummy_ext')
        
    return doc_extractions