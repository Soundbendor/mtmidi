import os
import polars as pl
import util as UM
import sys

wavdir = '/nfs/hpc/share/kwand/wav'
testdir = '/nfs/hpc/share/kwand/test'
dataset = 'polyrhythms'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

dswavdir = os.path.join(wavdir, dataset)
dstestdir = os.path.join(testdir, dataset)
csvdir = UM.by_projpath(subpath='csv',make_dir = False)
csvfile = os.path.join(csvdir, f'{dataset}-folds.csv')
df = pl.read_csv(csvfile)

os.mkdir(dstestdir)

for i in range(len(df)):
    cur_name = df[i]['name'][0]
    cur_wav = f'{cur_name}.wav'
    cur_wp = os.path.join(dswavdir, cur_wav)
    cur_file = f'{cur_name}.test'
    cur_fp = os.path.join(dstestdir, cur_file)
    if os.path.isfile(cur_wp) == False:
        print(f'{cur_file} not found')
        break
    else:
        file = open(cur_fp, 'w')
        file.close()
        


