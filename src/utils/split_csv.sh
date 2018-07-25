#!/bin/bash
FILE=$1
NUM=$2

FILEPATH="$(readlink -f ${FILE})"
EXTENSION="${FILE##*.}"
FILENAME="${FILE%.*}"

mkdir $FILENAME
cd $FILENAME

split -d --additional-suffix ".${EXTENSION}" -n $NUM $FILEPATH ${FILENAME}_shard_
