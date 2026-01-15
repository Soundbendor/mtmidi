import os
import polars as pl
import util as UM
import sys

mv_test = True
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

#os.mkdir(dstestdir)

for i in range(len(df)):
    cur_name = df[i]['name'][0]
    cur_wav = f'{cur_name}.wav'
    cur_wp = os.path.join(dswavdir, cur_wav)
    cur_test = f'{cur_name}.test'
    cur_tp = os.path.join(dstestdir, cur_test)
    cur_fold = df[i]['fold'][0]
    fold_folder = f'fold_{cur_fold}'
    print(cur_name, cur_fold)
    from_dir = None
    cur_file = None
    from_fp = None
    if mv_test == True:
        from_dir = dstestdir
        filename = cur_test
        from_fp = cur_tp
    else:
        from_dir = dswavdir
        filename = cur_wav
        from_fp = cur_wp

    fold_fp = os.path.join(from_dir, fold_folder)
    to_fp = os.path.join(fold_fp, filename)
    if os.path.isdir(fold_fp) == False:
        os.mkdir(fold_fp)
    os.rename(from_fp, to_fp)


