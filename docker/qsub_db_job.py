import argparse
import sys, os
parser = argparse.ArgumentParser(description='Submit DB Jobs.')
parser.add_argument('--input', required=True,
                    help='directory containing pre-parsed DBs')
parser.add_argument('--logdir', required=True, help='default log path')
parser.add_argument('--outdir', required=True, help='default output path')
parser.add_argument('--check', default=None)
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    print("Output saved to {}".format(args.outdir))

if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)
    print("Logs saved to {}".format(args.logdir))

jbs = 0

files = [os.path.join(root, name)
             for root, dirs, files in os.walk(args.input)
             for name in files
             if name.endswith("-db")]

for flpth in sorted(files):
    if flpth.endswith('-db'):
        fl = os.path.split(flpth)[-1]
        if args.check is not None:
            respath = os.path.join(args.outdir,'args.check',f'{args.check}_extraction_'+fl+'.jsonl')
        else:
            respath = '3!@#$%^^' #dummy path which will never be found
        #respath = os.path.join(args.outdir,'phone','phone_extraction_'+fl+'.jsonl')
        flpth = flpth
        if not os.path.exists(respath):
            cmd = "qsub -V -b y -r y -N job-{} -wd {} 'sh /dfs/scratch0/jdunnmon/chtap/extractors/docker/run_extractors_docker_local_dfs_fonduer.sh {} {}'".format(fl, args.logdir, flpth, args.outdir)
            os.system(cmd)
            jbs+=1
        else:
            print('Already done, skipping...')
        #if jbs >0:
        #    break
    else:
        pass

print('Started {} jobs...'.format(jbs))
