FILE=$1
FILENAME=$(basename $FILE)

docker run -v $FILE:/data/$FILENAME -v /lfs/local/0/jdunnmon/chtap/extractors/docker/run_config.json:/config/run_config.json -v /lfs/local/0/jdunnmon/chtap/extractors/docker/test_output_docker:/output jdunnmon/chtap-manual:06-02-2019-postgres11 bash /home/repos/extractors/docker/entry.sh
