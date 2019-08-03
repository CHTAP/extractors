INFILE=$1
CONFIG=$2
DBNAME=docker

#function contains() {
#    local n=$#
#    local value=${!n}
#    for ((i=1;i < $#;i++)) {
#        if [ "${!i}" == "${value}" ]; then
#            return 0
#        fi
#    }
#    return 1
#}

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
cd /dfs/scratch0/jdunnmon/chtap/extractors/src/shell
#sh ../src/utils/kill_db.sh ${DBNAME}; 

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
fi

# Running phone extractor
#echo "Running phone extractor..."
#python evaluate_phone_extractor.py -f ${INFILE} -c $CONFIG 
#echo "Phone extractor complete!"

# Running email extractor
#echo "Running email extractor..."
#python evaluate_email_extractor.py -f ${INFILE} -c $CONFIG
#echo "Email extractor complete!"

# Running age extractor
#echo "Running age extractor..."
#python evaluate_age_extractor.py -f ${INFILE} -c $CONFIG
#echo "Age extractor complete!"

# Running ethnicity extractor
#echo "Running ethnicity extractor..."
#python evaluate_ethnicity_extractor.py -f ${INFILE} -c $CONFIG
#echo "Ethnicity extractor complete!"

# TODO: FOR LOCATION: NEED TO ADD SAVED LSTM PATH!
# Running location extractor
#echo "Running location extractor..."
#python extract_location_candidates.py -f ${INFILE} -c $CONFIG
#python evaluate_location_extractor.py -f ${INFILE} -c $CONFIG
#echo "Location extractor complete!"

# Running incall outcall extractor
#echo "Running incall outcall extractor..."
#python evaluate_incall_outcall_extractor.py -f ${INFILE} -c $CONFIG
#echo "Location extractor complete!"

#Running price extractor
#echo "Running price per hour extractor..."
#python extract_price_candidates.py -f ${INFILE} -c $CONFIG
#python evaluate_price_extractor.py -f ${INFILE} -c $CONFIG -n hour
#echo "Price extractor complete!"
echo "Extractors complete!"
