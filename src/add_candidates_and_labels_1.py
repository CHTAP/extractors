import json
# Loading config
with open("run_config.json") as fl:
    cfg = json.load(fl)
cfg_params = cfg['parameters']

# Setting snorkel path and output root
import os
from os.path import join
output_root = join(cfg_params['output_path'],cfg_params['experiment_name'])
os.environ['FONDUERDBNAME'] = cfg_params['postgres_db_name']
os.environ['SNORKELDB'] = join(cfg_params['postgres_location'],os.environ['FONDUERDBNAME'])

# For loading input files
import pandas as pd

# For running Snorkel
from snorkel.contrib.fonduer import SnorkelSession
from snorkel.contrib.fonduer.models import candidate_subclass
from snorkel.contrib.fonduer import HTMLPreprocessor, OmniParser

if __name__ == "__main__":
    # Load labeled data from tsv
    pth_labeled = cfg_params['data_path']
    fl_labeled = cfg_params['labeled_data_file']
    df_labeled = pd.read_csv(os.path.join(pth_labeled,fl_labeled),sep='\t')

    # Start snorkel session and creating location subclass
    session = SnorkelSession()
    Location_Extraction = candidate_subclass('location_extraction',\
                          ["location"])

    # Parsing documents 
    max_docs = cfg_params['max_docs']
    doc_preprocessor = HTMLPreprocessor(pth_labeled, max_docs=max_docs)

    corpus_parser = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    %time corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    
    # TODO: Load unlabeled data from tsv

    # TODO: Parse labeled data into candidates, add ground truth labels, add to db

    # TODO: Parse unlabeled data into candidates, add to db
