#!/bin/bash

PTH=$1
STARTIND=$2
ENDIND=$3

NODES=("dawn1" "dawn2" "dawn3" "dawn4" "dawn5" "dawn6" "dawn7"
       "dawn8" "dawn9" "dawn10" "dawn11" "dawn12" "dawn13"
       "dawn14" "dawn15")
 
#NODES=("dawn1" "dawn2" "dawn3") 
FOLDERS=(`find ${PTH} -mindepth 1 -maxdepth 1 -type d`)

for iter in `seq $STARTIND $ENDIND`
do  
    NODE_INDEX=$(($iter-$STARTIND))
    echo $NODE_INDEX
    DBNAME=`basename ${FOLDERS[$iter]}`
    echo "Database: $DBNAME"
    echo "Starting node ${NODES[NODE_INDEX]} with database ${DBNAME}"
    ssh -t jdunnmon@${NODES[NODE_INDEX]} <<< "export SNORKELDB=postgresql://jdunnmon:123@localhost:5432/${DBNAME}; export CUDA_VISIBLE_DEVICES=0; cd /dfs/scratch1/jdunnmon/repos/extractors/src/shell; source activate snorkel; echo 'Running price candidate extraction...'; python extract_price_candidates.py -f ${FOLDERS[$iter]}; echo 'Evaluating price extractor for node ${NODES[NODE_INDEX]} with database ${DBNAME} ...'; python evaluate_price_extractor.py -f ${FOLDERS[$iter]}; bash -l" &
done