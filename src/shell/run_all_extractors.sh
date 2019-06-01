#!/bin/bash

PTH=$1

bash run_location_extractor.sh $PTH
bash run_price_extractor.sh $PTH
bash run_phone_extractor.sh $PTH
bash run_email_extractor.sh $PTH
