#!/bin/bash bash
source activate chtap
echo "Entry point activated!"

echo "Moving to extractor working directory"
cd /home/repos/extractors/docker

echo "Pulling repo"
git pull origin fonduer_extractors

echo "Starting postgres"
sudo pg_createcluster 11 main
sudo /etc/init.d/postgresql start

echo "Creating user"
sudo -u postgres -H -- psql -c "CREATE USER docker WITH PASSWORD 'docker'; ALTER USER docker WITH SUPERUSER;"
sudo -u postgres -H -- psql -c "CREATE DATABASE docker;"

# PUT CODE HERE FOR RUNNING EXTRACTORS!
for entry in "/data"/*
do
echo "$entry"
if [ "$entry" != "/data/input.csv" ]; then
    echo "Running extractors on $entry"
    bash /home/repos/extractors/docker/run_extractors.sh $entry /config/run_config.json
fi
done


# COMPLETE!
echo "Extraction complete"
