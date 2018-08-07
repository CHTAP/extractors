#!/bin/bash

PTH=$1
NODES=("raiders5" "raiders2")
FOLDERS=(`find ${PTH} -mindepth 1 -maxdepth 1 -type d`)

for iter in `seq 0 $((${#NODES[@]}-1))`
do  
    DBNAME=`basename ${FOLDERS[$iter]}`
    echo "Database: $DBNAME"
    echo "Starting node ${NODES[$iter]} with database ${DBNAME}"
    ssh -t jdunnmon@${NODES[$iter]} <<< "export SNORKELDB=postgresql://jdunnmon:123@localhost:5432/${DBNAME}; export CUDA_VISIBLE_DEVICES=0; cd /dfs/scratch1/jdunnmon/repos/extractors/src/shell; source activate snorkel; echo 'Evaluating email extractor...'; python evaluate_email_extractor.py -f ${FOLDERS[$iter]}; bash -l" &
done
