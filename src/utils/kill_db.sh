#!/usr/bin/env bash
# kill all connections to the postgres server
if [ -n "$1" ] ; then
  where="where pg_stat_activity.datname = '$1'"
  echo "killing all connections to database '$1'"
else
  echo "killing all connections to database"
fi

cat <<-EOF | psql -U jdunnmon -d postgres 
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
${where}
EOF

echo "reinitializing database"
dropdb $1
createdb $1
