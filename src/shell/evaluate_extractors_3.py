import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--extraction_field','-e',type=str,default='all')
parser.add_argument('--index','-i',type=str,default='chtap')
parser.add_argument('--max_docs', '-m', type=int, default=10000)
parser.add_argument('--out_fields', '-of', type=str, default='full')
parser.add_argument('--terms', '-t', type=str, default='')
parser.add_argument('--parallel','-p',type=int, default=1)
parser.add_argument('--suffix','-s',type=str,default='')
args = parser.parse_args()



