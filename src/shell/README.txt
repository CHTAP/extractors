STEPS TO RUN EXTRACTORS IN DISTRIBUTED FASHION

#1: call sample_es_query_dsl.py with threads = number of slices desired
#2: call preprocess_tsvs_0.py on elastic search output to preprocess raw content into `parsed` subdirectory
#3: call shard_tsvs_1.sh on `parsed` subdirectory to create shards for each parallel db read
#4: call create_dbs_2.sh using appropriate machine-file mapping to create databases for each file, write database names on each machine to text file
#5: call extract_candidates_3.sh using machine-db mapping to extract candidates on for all dbs on each machine
#6: call evaluate_extractors_4.sh on all dbs on each machine, write outputs to common dfs repository
#7: call cleanup_dbs_5.sh to remove dbs on all client machines 
