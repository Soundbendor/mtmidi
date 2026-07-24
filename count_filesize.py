import librosa as lr
import os
import polars as pl

#ds = 'polyrhythms'
ds = 'dynamics'

home_dir = '/nfs/hpc/share/kwand/syntheory_plus'

load_dir = os.path.join(home_dir, ds)

def get_dynamics_rvb_offset(name):
    cur = fname.split('_')
    _rvb = int(cur[-2][3])
    _off = int(cur[-1][3:])
    return (_rvb, _off)

def get_polyrhythms_bpm_rvb_offset(name):
    cur = fname.split('-')
    inst = cur[0]
    bpm = cur[1]
    rvb = cur[2]
    offset = cur[3]
    tup = tuple([int(x.split('_')[1]) for x in cur[1:4]])
    return tup

fnames = []
num_samp = []
sr = []
bpm = []
rvblvl = []
offset = []
for f in os.listdir(load_dir):
    fname = f.split('.')[0]
    fp = os.path.join(load_dir, f)
    x, _sr = lr.load(fp, sr=None, mono=False)
    fnames.append(fname)
    num_samp.append(x.shape[-1])
    sr.append(_sr)
    cur = None
    if ds == 'polyrhythms':
        cur = get_polyrhythms_bpm_rvb_offset(fname)
        bpm.append(cur[0])
        rvblvl.append(cur[1])
        offset.append(cur[2])
    else:
        cur = get_dynamics_rvb_offset(fname)
        rvblvl.append(cur[0])
        offset.append(cur[1])

data = None
schema = None
if ds == 'polyrhythms':
    data = {'name': fnames, 'bpm': bpm, 'rvb': rvblvl, 'offset': offset, 'num_samp': num_samp, 'sr': sr}
    schema = [('name', pl.String), ('bpm', pl.Int64), ('rvb', pl.Int64), ('offset', pl.Int64), ('num_samp', pl.Int64), ('sr', pl.Int64)]
else:
    data = {'name': fnames, 'rvb': rvblvl, 'offset': offset, 'num_samp': num_samp, 'sr': sr}
    schema = [('name', pl.String), ('rvb', pl.Int64), ('offset', pl.Int64), ('num_samp', pl.Int64), ('sr', pl.Int64)]

df = pl.DataFrame(data, schema=schema)
df.write_csv(f'{ds}-counts.csv')
#print(df)

