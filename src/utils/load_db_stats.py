import argparse
from collections import defaultdict
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Get DB stats.')
    parser.add_argument('--input', required=True,
                    help='input file')
    parser.add_argument('--output', required=False, help='default output path',
                        default=None)
    args = parser.parse_args()
    return args


def main(args):
   
    counts_dict = defaultdict(list)

    with open(args.input,'r') as fl:
        counts_data = json.load(fl)
 
    for ky,val in counts_data.items():
        year, month, day, _ = ky.split('-')
        counts_dict[month].append(int(val))

    kys = list(counts_dict.keys())
    kys.sort()

    val_tot = 0
    for ky in kys:
        val = np.sum(counts_dict[ky])
        print(f'Number of Ads in {ky} {year}: {val}')
        val_tot += val
   
    print('\n') 
    print(f'Total Number of Ads in {year}: {val_tot}')
     
if __name__=="__main__":
    args = parse_args()
    main(args)
