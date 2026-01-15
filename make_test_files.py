import os
import polars as pl
import util as UM
import sys

wavdir = '/nfs/hpc/share/kwand/wav'

dataset = 'polyrhythms'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

dswavdir = os.path.join(wavdir, dataset)
csvdir = UM.by_projpath(subpath='csv',make_dir = False)
csvfile = os.path.join(csvdir, f'{dataset}-folds.csv')
df = pl.read_csv(csvfile)
for i in range(len(df)):
    cur_name = df[i]['name'][0]
    cur_file = f'{cur_name}.wav'
    cur_fp = os.path.join(dswavdir, cur_file)
    if os.path.isfile(cur_fp) == False:
        print(f'{cur_file} not found')
        break
