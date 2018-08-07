PTH=$1
NUM=$2

bash create_create_dbs_2.sh $PTH $NUM
bash run_location_extractor.sh $PTH
bash run_price_extractor.sh $PTH
bash run_phone_extractor.sh $PTH
bash run_email_extractor.sh $PTH