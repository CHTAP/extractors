import json
# Loading config
with open("run_config.json") as fl:
    cfg = json.load(fl)
cfg_params = cfg['parameters']

# Setting snorkel path and output root
import os
from os.path import join
output_root = join(cfg_params['output_path'],cfg_params['experiment_name'])
os.environ['SNORKELDB'] = '{0}:///{1}/{2}'.format("sqlite", join(output_root,'data'), "snorkel_exp.db")

import pandas as pd

if __name__ == "__main__":
    # Load labeled data from tsv
    pth_labeled = cfg_params['data_path']
    fl_labeled = cfg_params['labeled_data_file']
    df_labeled = pd.read_csv(os.path.join(pth_labeled,fl_labeled),sep='\t')

    # TODO: Load unlabeled data from tsv

    # TODO: Parse labeled data into candidates, add ground truth labels, add to db

    # TODO: Parse unlabeled data into candidates, add to db
