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
