import sys
import os
import csv

def shard(file_loc, num=2):
    """
    Creates new tsv file by splitting given tsv.
    
    string file_loc: path to data source -- .tsv file
    int num: number of files to create

    """
    
    csv.field_size_limit(sys.maxsize)
    
    num = int(num)
    file_name = os.path.basename(file_loc)
    file_name_base = file_name.replace('.tsv', '')
    out_dir = os.getcwd() + '/' + file_name_base
    out_loc_base = out_dir + '/' + file_name_base
    os.makedirs(out_dir)

    with open(file_loc, 'r') as in_file:
        tsv_reader = csv.reader(in_file, delimiter='\t')
        lines = list(tsv_reader)
        num_rows = len(lines)
        rows_per_file = num_rows // num
        
        for i in range(num):
            print(f'Writing shard {i} of {num}...')
            with open(out_loc_base + '_' + str(i) + '.tsv', 'w') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                for line in lines[rows_per_file*i:rows_per_file*(i+1)]:
                    writer.writerow(line)
                if i == num - 1:
                    for line in lines[rows_per_file*(i+1):]:
                        writer.writerow(line)

def main():
    if len(sys.argv) != 3:
        print("Needs two args: file to shard and number of shards.")
    else:
        shard(sys.argv[1], sys.argv[2])

if __name__== '__main__':
    main()
