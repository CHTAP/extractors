#!/bin/bash bash
# Adding
UN=$1
sudo adduser -u $UN memexuser
sudo passwd -d memexuser
sudo usermod -aG sudo memexuser

sudo su - memexuser

source activate chtap
echo "Entry point activated!"

#if  [ "$(id -un)" == "root" ]; then
#	   echo "This script should not be run as root! Use -u option!" 
#	      exit 1
#fi

echo "Moving to extractor working directory"
cd /code/docker

echo "Starting postgres"
sudo pg_createcluster 11 main
sudo /etc/init.d/postgresql start

echo "Creating user"
sudo -u postgres -H -- psql -c "CREATE USER docker WITH PASSWORD 'docker'; ALTER USER docker WITH SUPERUSER;"
sudo -u postgres -H -- psql -c "CREATE DATABASE docker;"

# PUT CODE HERE FOR RUNNING EXTRACTORS!
echo "Running extractor as id ${UN}"
#sudo su - memexuser

for entry in "/data"/*
do
echo "$entry"
if [ "$entry" != "/data/input.csv" ]; then
    echo "Running extractors on $entry"
    bash /code/docker/run_extractors_v2.sh $entry /config/run_config.json
    sudo chmod -R 777 /output
fi
done


# COMPLETE!
echo "Extraction complete"
