#!/bin/bash

DIRECTORY=$1
NUMBER=$2

for entry in `ls $DIRECTORY`; do
   echo "Sharding file $entry"
   bash ../utils/split_csv.sh "${DIRECTORY}/${entry}" $NUMBER
done
