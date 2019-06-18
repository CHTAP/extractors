import argparse
import os
from collections import Counter

parser = argparse.ArgumentParser(description='Check for empty extractions')
parser.add_argument('--pdir', help="parent directory for extractions", required=True)
args = parser.parse_args()

pdir = args.pdir
dirlist = os.listdir(pdir)
all_rerun_files = []
for dr in dirlist:
    total_files = [os.path.join(pdir,dr,a) for a in os.listdir(os.path.join(pdir,dr))]
    empty_files = [a for a in total_files if os.path.getsize(a)==0]
    all_rerun_files += empty_files
    print(f"Empty files for {dr} extraction: {len(empty_files)} out of {len(total_files)}")

all_rerun_dbs = [a.split('.')[0].split('/')[-1].split('_')[-1] for a in all_rerun_files]
all_rerun_dbs = list(set(all_rerun_dbs))
all_rerun_dbs_years = [a.split('-')[0] for a in all_rerun_dbs]

print(f"Total data files to rerun: {len(all_rerun_dbs)}")
for yr, cnt in Counter(all_rerun_dbs_years).items():
    print(f"Files to rerun from {yr}: {cnt}")
