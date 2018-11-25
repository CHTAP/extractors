INFILE=$1
CONFIG=$2
DBNAME=docker

# Exporting CUDA env
CUDA_VISIBLE_DEVICES=0

# Initializing database and moving to container home
#echo "Creating Database: $DBNAME" 
cd /home/repos/extractors/src/shell
#sh ../src/utils/kill_db.sh ${DBNAME}; 

# Activating environment
source activate chtap 

# Creating database
echo "Filling Database: $DBNAME" 
python create_db_docker.py -f ${INFILE} -c $CONFIG
echo "Completed database creation!"

# Running phone extractor
echo "Running phone extractor..."
python evaluate_phone_extractor.py -f ${INFILE} -c $CONFIG 
echo "Phone extractor copmlete!"
