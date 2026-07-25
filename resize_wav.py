import polars as pl
import librosa as lr
import soundfile as sf
import os 

pldf = pl.read_csv('polyrhythms-counts.csv')
ddf = pl.read_csv('dynamics-counts.csv')

#cur_wav_dir = '/home/dxk/osu/mtmidi/wav'
#out_wav_dir = '/home/dxk/osu/mtmidi/wav_out'
cur_wav_dir = '/nfs/hpc/share/kwand/syntheory_plus/'
out_wav_dir = '/nfs/hpc/share/kwand/syntheory_plus/'
dses = ['polyrhythms', 'dynamics']
def polyrhythms_get_offset(name):
    cur_name = name.split('-')
    offms_substr = cur_name[-2]
    cur_offset = int(offms_substr.split('_')[1])
    return cur_offset

def polyrhythms_get_zero_offset_name(name):
    cur_name = name.split('-')
    cur_name[-2] = 'offms_0'
    ret = '-'.join(cur_name)
    return ret

def dynamics_get_offset(name):
    cur_name = name.split('_')
    offset_substr = cur_name[-1]
    offset = int(offset_substr[3:])
    return offset

def dynamics_get_zero_offset_name(name):
    cur_name = name.split('_')
    cur_name[-1] = 'off0'
    ret = '_'.join(cur_name)
    return ret

def get_num_samp(df, cur_name):
    cur_entry = df.filter(pl.col('name') == cur_name)[0]
    #cur_num_samp = 0
    cur_num_samp = cur_entry['num_samp']
    return cur_num_samp[0]

for ds in dses:
    cur_wav_path = os.path.join(cur_wav_dir, ds)
    cur_out_path = os.path.join(out_wav_dir, ds)
    for f in os.listdir(cur_wav_path):
        cur_path = os.path.join(cur_wav_path, f)
        out_path = os.path.join(cur_out_path, f)
        cur_name = f.split('.')[0]
        cur_offset = -1
        if ds == 'polyrhythms':
            cur_offset = polyrhythms_get_offset(cur_name)
            cur_df = pldf
        else:
            cur_offset = dynamics_get_offset(cur_name)
            cur_df = ddf
        cur_samp = get_num_samp(cur_df, cur_name)
        if cur_offset > 0:
            cur_zero_name = None
            if ds == 'polyrhythms':
                cur_zero = polyrhythms_get_zero_offset_name(cur_name)
            else:
                cur_zero = dynamics_get_zero_offset_name(cur_name)

            cur_no_samp = get_num_samp(cur_df, cur_zero)


            if cur_samp != cur_no_samp:
                cur_dat, cur_sr = sf.read(cur_path)
                new_dat = cur_dat[:cur_no_samp]
                #print(cur_dat.shape, new_dat.shape, cur_dat.dtype, cur_samp, cur_no_samp)
                print(f'writing {cur_name} cutting down to {cur_no_samp} from {cur_samp}')
                sf.write(out_path, new_dat, cur_sr, 'FLOAT')
    """
for row in pldf.iter_rows(named=True):
    cur_name = row['name']
    cur_offset = polyrhythms_get_offset(cur_name)
    cur_samp = get_num_samp(pldf, cur_name)
    cur_zero = polyrhythms_get_zero_offset_name(cur_name)
    cur_no_samp = get_num_samp(pldf, cur_zero)

    print(cur_name, cur_offset, cur_samp, cur_no_samp)

"""
"""
for row in pldf.iter_rows(named=True):
    cur_name = row['name']
    cur_offset = polyrhythms_get_offset(cur_name)
    cur_zero = polyrhythms_get_zero_offset_name(cur_name)
    cur_samp = get_num_samp(pldf, cur_name)
    cur_no_samp = get_num_samp(pldf, cur_zero)
    print(cur_name, cur_offset, cur_samp, cur_no_samp)

for row in ddf.iter_rows(named=True):
    cur_name = row['name']
    cur_offset = dynamics_get_offset(cur_name)
    cur_zero = dynamics_get_zero_offset_name(cur_name)
    cur_samp = get_num_samp(ddf, cur_name)
    cur_no_samp = get_num_samp(ddf, cur_zero)
    print(cur_name, cur_offset, cur_samp, cur_no_samp)

"""
