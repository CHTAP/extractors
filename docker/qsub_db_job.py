import argparse
import sys, os
parser = argparse.ArgumentParser(description='Submit DB Jobs.')
parser.add_argument('--input', required=True,
                    help='directory containing pre-parsed DBs')
parser.add_argument('--logdir', required=True, help='default log path')
parser.add_argument('--outdir', required=True, help='default output path')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print("Output saved to {}".format(args.outdir))

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
    print("Logs saved to {}".format(args.logdir))

jbs = 0
for fl in sorted(os.listdir(args.input)):
    if fl.endswith('-db'):
        respath = os.path.join(args.outdir,'price_per_hour','price_per_hour_extraction_'+fl+'.jsonl')
        flpth = os.path.join(args.input,fl)
        if True:
        #if not os.path.exists(respath):
            cmd = "qsub -V -b y -r y -N job-{} -wd {} 'sh /dfs/scratch0/jdunnmon/chtap/extractors/docker/run_extractors_docker_local_dfs_fonduer.sh {} {}'".format(fl, args.logdir, flpth, args.outdir)
            os.system(cmd)
            jbs+=1
        else:
            print('Already done, skipping...')
        if jbs >5:
            break
    else:
        pass

print('Started {} jobs...'.format(jbs))
