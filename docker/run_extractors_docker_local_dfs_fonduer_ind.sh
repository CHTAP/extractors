FILE=$1
OUTDIR=$2

FILENAME=$(basename $FILE)

#chmod 666 /var/run/docker.sock

#systemctl start docker
#docker container prune
#docker volume prune
docker pull jdunnmon/chtap-manual:06-02-2019-postgres11


docker run --rm --shm-size='8gb' -v /dfs/scratch0/jdunnmon/chtap/extractors:/code -v $FILE:/data/$FILENAME -v /dfs/scratch0/jdunnmon/chtap/extractors/docker/run_config.json:/config/run_config.json -v $OUTDIR:/output jdunnmon/chtap-manual:06-02-2019-postgres11 bash /code/docker/entry_v2.sh

#docker run -v $FILE:/data/$FILENAME -v /lfs/local/0/jdunnmon/chtap/extractors/docker/run_config.json:/config/run_config.json jdunnmon/chtap-manual:latest tail -f /dev/null