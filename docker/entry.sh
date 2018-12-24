#!/bin/bash bash
source activate chtap
echo "Entry point activated!"

echo "Moving to extractor working directory"
cd /home/repos/extractors/docker

echo "Pulling repo"
git pull origin dist_exec_docker

echo "Starting postgres"
sudo /etc/init.d/postgresql restart

# PUT CODE HERE FOR RUNNING EXTRACTORS!
for entry in "/data"/*
do
    echo "Running extractors on $entry"
    bash /home/repos/extractors/docker/run_extractors.sh $entry /config/run_config.json
done

# COMPLETE!
echo "Extraction complete"
