INFILE=$1
CONFIG=$2
DBNAME=docker

# Initializing database and moving to container home
#echo "Creating Database: $DBNAME" 
cd /home/repos/extractors/docker
#sh ../src/utils/kill_db.sh ${DBNAME}; 

# Activating environment
source activate chtap 

# Creating database
echo "Filling Database: $DBNAME" 
python create_db.py -f ${INFILE} -c $CONFIG -d $DBNAME
echo "Completed database creation!"

