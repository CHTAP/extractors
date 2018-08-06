#!/bin/bash

PTH=$1
NODES=("raiders5" "raiders2")
FOLDERS=(`find ${PTH} -mindepth 1 -maxdepth 1 -type d`)

for iter in `seq 0 $((${#NODES[@]}-1))`
do  
    DBNAME=`basename ${FOLDERS[$iter]}`
    echo "Database: $DBNAME"
    echo "Starting node ${NODES[$iter]} with folder ${FOLDERS[$iter]}"
    ssh -t jdunnmon@${NODES[$iter]} <<< "export SNORKELDB=postgresql://jdunnmon:123@localhost:5432/${DBNAME}; cd /dfs/scratch1/jdunnmon/repos/extractors/src/shell; sh ../utils/kill_db.sh ${DBNAME}; source activate snorkel; python create_dbs.py -f ${FOLDERS[$iter]}; bash -l" & | tee -a output_dbs.txt
done

