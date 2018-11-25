#!/bin/bash bash
source activate chtap
echo "Entry point activated!"
# PUT CODE HERE FOR RUNNING EXTRACTORS!
bash run_extractors.sh /data/input.csv /home/repos/extractors/docker/run_config.json

