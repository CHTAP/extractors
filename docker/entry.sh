#!/bin/bash bash
source activate chtap
echo "Entry point activated!"

echo "Pulling repo"
git pull origin dist_exec_docker

echo "Starting postgres"
sudo /etc/init.d/postgresql restart

# PUT CODE HERE FOR RUNNING EXTRACTORS!
echo "Running extractors"
bash /home/repos/extractors/docker/run_extractors.sh /data/input.csv /home/repos/extractors/docker/run_config.json

