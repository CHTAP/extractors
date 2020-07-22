
# coding: utf-8

# # Step 1: Build the Dataset

# The first thing to do is ensure that modules are auto-reloaded at runtime to allow for development in other files.

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# We then set the Snorkel database location and start and connect to it.  By default, we use a PosgreSQL database backend, which can be created using `createdb DB_NAME` once psql is installed.  Note that Snorkel does *not* currently support parallel database processing with a SQLite backend.

# In[6]:


# Setting Snorkel DB location
import os
import sys

import random
import numpy as np

#For networked PostgreSQL
postgres_location = 'postgresql://jdufault:123@localhost:5432'
postgres_db_name = 'es_locs_1M'
os.environ['SNORKELDB'] = os.path.join(postgres_location,postgres_db_name)

#For local PostgreSQL
#os.environ['SNORKELDB'] = 'postgres:///es_locs_small'

# Adding path above for utils
sys.path.append('../utils')

# For SQLite
#db_location = '.'
#db_name = "es_locs_small.db"
#os.environ['SNORKELDB'] = '{0}:///{1}/{2}'.format("sqlite", db_location, db_name)

# Start Snorkel session
from snorkel import SnorkelSession
session = SnorkelSession()

# Setting parallelism
parallelism = 32

# Setting random seed
seed = 1701
random.seed(seed)
np.random.seed(seed)


# We now set the document preprocessor to read raw data into the Snorkel database.  There exist three possible data source options: JSONL files from the MEMEX project (option: `memex_jsons`), a raw tsv file of extractions from the memex project `content.tsv` (option: `content.tsv`), and tsvs with a similar format to `content.tsv` drawn from an Elasticsearch index of the data (option: `es`).  `max_docs` controls the number of documents read by the preprocessor, and `data_source` sets the location of the data.  For MEMEX json source, this should be a directory, while in all other cases it should be a tsv file.

# In[7]:


from dataset_utils import set_preprocessor, combine_dedupe

# Set data source: options are 'content.tsv', 'memex_jsons', 'es'
data_source = 'es'

# Setting max number of docs to ingest
max_docs = 100

# Setting location of data source

# For ES:
data_loc = '/lfs/raiders5/0/jdunnmon/data/chtap/output_all'

# Optional: add tsv with additional documents to create combined tsv without duplicates
#data_loc = combine_dedupe(data_loc, 'output_location.tsv', 'combined.tsv')

# If memex_raw_content is a content_field, uses term as a regex in raw data in addition to getting title and body
term = r'\b[Ll]ocation:|\b[cC]ity:'

# Doc length in characters, remove to have no max
max_doc_length=1500

# Setting preprocessor
doc_preprocessor = set_preprocessor(data_source, data_loc, max_docs=max_docs, verbose=False, clean_docs=True,
                                    content_fields=['raw_content', 'url'], term=term, max_doc_length=max_doc_length)


# Now, we execute the preprocessor.  Parallelism can be changed using the `parallelism` flag.  Note that we use the Spacy parser rather than CoreNLP, as this tends to give superior results.

# In[ ]:


from snorkel.parser import CorpusParser
from snorkel.parser.spacy_parser import Spacy

# Applying corpus parser
corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(list(doc_preprocessor), parallelism=parallelism, verbose=False)


# Checking the number of parsed documents and sentences in the database.

# In[ ]:


from snorkel.models import Document, Sentence

# Printing number of docs/sentences
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())


# Separating into train, dev, and test sets

# In[11]:


from dataset_utils import create_test_train_splits

# Getting all documents parsed by Snorkel
docs = session.query(Document).order_by(Document.name).all()

# Creating train, test, dev splits
train_docs, dev_docs, test_docs, train_sents, dev_sents, test_sents = create_test_train_splits(docs, 'location', gold_dict=None, dev_frac=0.1, test_frac=0.1)

# Create candidate extractor.

# In[12]:


from snorkel.candidates import Ngrams
from snorkel.candidates import CandidateExtractor
from dataset_utils import create_candidate_class, LocationMatcher, fast_loc
from snorkel.matchers import Union, LambdaFunctionMatcher

# Setting extraction type -- should be a subfield in your data source extractions field!
extraction_type = 'location'

# Creating candidate class
candidate_class, candidate_class_name = create_candidate_class(extraction_type)

# Defining ngrams for candidates
location_ngrams = Ngrams(n_max=3)

# Define matchers
geotext_location_matcher = LambdaFunctionMatcher(func=fast_loc)
spacy_location_matcher = LocationMatcher(longest_match_only=True)

# Union matchers and create candidate extractor
location_matcher = Union(geotext_location_matcher)
cand_extractor   = CandidateExtractor(candidate_class, [location_ngrams], [location_matcher])


# Applying candidate extractor to each split (train, dev, test)

# In[ ]:


# Applying candidate extractor to each split
for k, sents in enumerate([train_sents, dev_sents, test_sents]):
    cand_extractor.apply(sents, split=k, parallelism=parallelism)
    print("Number of candidates:", session.query(candidate_class).filter(candidate_class.split == k).count())


# Add gold labels.

# In[ ]:


from dataset_utils import get_gold_labels_from_meta


# Adding dev gold labels using dictionary
missed_dev = get_gold_labels_from_meta(session, candidate_class, extraction_type, 1, annotator='gold', gold_dict=None)

# Adding test gold labels using dictionary
missed_test = get_gold_labels_from_meta(session, candidate_class, extraction_type, 2, annotator='gold', gold_dict=None)

# In[ ]:


# Checking percent of gold labels that are positive
from dataset_utils import check_gold_perc
perc_pos = check_gold_perc(session)


# In[ ]:


from dataset_utils import remove_gold_labels
# Remove gold labels if you want -- uncomment!
#remove_gold_labels(session)

