INFILE=$1
CONFIG=$2
DBNAME=docker

function contains() {
    string="$1"
    substring="$2"
    if test "${string#*$substring}" != "$string"
    then
        return 0    # $substring is in $string
    else
        return 1    # $substring is not in $string
    fi
}

# Exporting CUDA env
CUDA_VISIBLE_DEVICES=0

# Initializing database and moving to container home
#echo "Creating Database: $DBNAME" 
#cd /home/repos/extractors/src/shell
#cd /lfs/raiders5/0/jdunnmon/chtap/extractors/docker
EXTRACTORS=($(./jq -r '.extractors_to_run' $CONFIG))
#cd /dfs/scratch0/jdunnmon/chtap/extractors/src/shell
#sh ../src/utils/kill_db.sh ${DBNAME}; 
cd /code/src/shell

# Activating environment
#source activate chtap 

# Creating database
if contains "${EXTRACTORS}" "create_db"; then
    echo "Filling Database: $DBNAME" 
    python create_db_docker.py -f ${INFILE} -c $CONFIG
    echo "Completed database creation!"
else
    echo "Skipping database creation..."
fi

# Running count extractor
if contains "${EXTRACTORS}" "count"; then
    echo "Running count extractor..." 
    python evaluate_count_extractor.py -f ${INFILE} -c $CONFIG
    echo "Count extractor complete!"
else
    echo "Skipping count extractor..."
fi

# Running domain extractor
if contains "${EXTRACTORS}" "domain"; then
    echo "Running domain extractor..." 
    python evaluate_domain_extractor.py -f ${INFILE} -c $CONFIG
    echo "Domain extractor complete!"
else
    echo "Skipping domain extractor..."
fi

# Running date extractor
if contains "${EXTRACTORS}" "date"; then
    echo "Running date extractor..." 
    python evaluate_date_extractor.py -f ${INFILE} -c $CONFIG
    echo "Date extractor complete!"
else
    echo "Skipping date extractor..."
fi

# Running time extractor
if contains "${EXTRACTORS}" "time"; then
    echo "Running time extractor..." 
    python evaluate_time_extractor.py -f ${INFILE} -c $CONFIG
    echo "Time extractor complete!"
else
    echo "Skipping time extractor..."
fi

# Running phone extractor
if contains "${EXTRACTORS}" "phone"; then
    echo "Running phone extractor..."
    python evaluate_phone_extractor.py -f ${INFILE} -c $CONFIG 
    echo "Phone extractor complete!"
else 
    echo "Skipping phone extractor..."
fi

# Running email extractor
if contains "${EXTRACTORS}" "email"; then
    echo "Running email extractor..."
    python evaluate_email_extractor.py -f ${INFILE} -c $CONFIG
    echo "Email extractor complete!"
else
    echo "Skipping email extractor..."
fi

# Running age extractor
if contains "${EXTRACTORS}" "age"; then
    echo "Running age extractor..."
    python evaluate_age_extractor.py -f ${INFILE} -c $CONFIG
    echo "Age extractor complete!"
else
    echo "Skipping age extractor..."
fi

# Running ethnicity extractor
if contains "${EXTRACTORS}" "ethnicity"; then    
    echo "Running ethnicity extractor..."
    python evaluate_ethnicity_extractor.py -f ${INFILE} -c $CONFIG
    echo "Ethnicity extractor complete!"
else
    echo "Skipping ethnicity extractor..."
fi

# TODO: FOR LOCATION: NEED TO ADD SAVED LSTM PATH!
# Running location extractor
if contains "${EXTRACTORS}" "location"; then
    echo "Running location extractor..."
    python extract_location_candidates.py -f ${INFILE} -c $CONFIG
    python evaluate_location_extractor.py -f ${INFILE} -c $CONFIG
    echo "Location extractor complete!"
else
    echo "Skipping location extractor..."
fi

# Running incall outcall extractor
if contains "${EXTRACTORS}" "incall_outcall"; then
    echo "Running incall outcall extractor..."
    python evaluate_incall_outcall_extractor.py -f ${INFILE} -c $CONFIG
    echo "Incall outcall extractor complete!"
else
    echo "Skipping incall outcall extractor..."
fi

#Running price extractor
if contains "${EXTRACTORS}" "price"; then
    echo "Running price per hour extractor..."
    python extract_price_candidates.py -f ${INFILE} -c $CONFIG
    python evaluate_price_extractor.py -f ${INFILE} -c $CONFIG -n hour
    echo "Price extractor complete!"
else
    echo "Skipping price extractor..."
fi

# End message
echo "Extractors complete!"
