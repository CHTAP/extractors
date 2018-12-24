FILE=$1
FILENAME=$(basename $FILE)

docker run -v $FILE:/data/$FILENAME -v /lfs/local/0/jdunnmon/test/run_config.json:/config/run_config.json -v /lfs/local/0/jdunnmon/test:/output jdunnmon/chtap-manual:latest bash /home/repos/extractors/docker/entry.sh
