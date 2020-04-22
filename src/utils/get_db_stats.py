import argparse
import sys, os
import psycopg2
import json
from fonduer import Meta
from fonduer.parser.models import Document

parser = argparse.ArgumentParser(description='Get DB stats.')
parser.add_argument('--input', required=True,
                    help='directory containing dbs')
parser.add_argument('--out_name', required=True, help='default output path')
args = parser.parse_args()

if __name__=="__main__":

    months = os.listdir(args.input)
    doc_num = {}
    for month in months:
        pth = os.path.join(args.input, month)
        db_files = os.listdir(pth)
        conn_string = 'postgresql://jdunnmon:123@localhost:5432/memex'
        for ii, fl in enumerate(db_files):
            print(f"Running file {fl}")
            with psycopg2.connect(database="jdunnmon", user="jdunnmon", password="123") as conn:
                with conn.cursor() as cur:
                    conn.autocommit = True
                    cur.execute("SELECT pg_terminate_backend (pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'memex'")
                    cur.execute('DROP DATABASE IF EXISTS memex')
                    cur.execute('CREATE DATABASE memex')
            os.system(f"psql memex < {os.path.join(pth,fl)}")
            session = Meta.init(conn_string).Session()
            docs = session.query(Document).count()
            print("==============================")
            print(f"DB contents for memex:")
            print(f'Number of documents: {session.query(Document).count()}')
            print("==============================")
            doc_num[fl] = docs

    with open(args.out_name,"w") as fl:
        json.dump(doc_num, fl)
