import argparse
import sys, os
parser = argparse.ArgumentParser(description='Submit CSV Jobs.')
parser.add_argument('--input', required=True,
                    help='directory containing preprocessed CSVs')
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
    if fl.endswith('.csv'):
        flpart = fl.split('.')[0]
        respath = os.path.join(args.outdir,'price_per_hour','price_per_hour_extraction_'+flpart+'.jsonl')
      #  wd = os.path.join(args.logdir,fl.split('.')[0])
      #  if not os.path.exists(wd):
      #      os.makedirs(wd)
        flpth = os.path.join(args.input,fl)
#        cmd = "qsub -V -b y -wd {} 'sudo sh /data/scripts/run_extractors_docker.sh {}'".format(wd, flpth)
        if not os.path.exists(respath):
#            cmd = "qsub -V -b y -r y -N {} -wd {} 'cp /data/scripts/run_extractors_docker.sh .; sudo sh run_extractors_docker.sh {} {}'".format(fl, args.logdir, flpth, args.outdir)
            cmd = "qsub -V -b y -r y -N {} -wd {} 'sh /dfs/scratch0/jdunnmon/chtap/extractors/docker/run_extractors_docker_local_dfs.sh {} {}'".format(fl, args.logdir, flpth, args.outdir)
            os.system(cmd)
            jbs+=1
        else:
            print('Already done, skipping...')
        if jbs >400:
            break
    else:
        pass

print('Started {} jobs...'.format(jbs))
