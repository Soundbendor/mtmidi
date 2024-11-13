import util as um
import musicnoteconv as mnc
from itertools import combinations
import csv
import random
import os
import polars as pl

data_profile = True
data_csv = True
data_png = True
harder_split = True
out_name = 'polyrhy_split1'

if harder_split == True:
    out_name = "polyrhy_split2"


out_csv = f'{out_name}.csv'
out_dir = "csv"
dp_dir = 'dataprof'
inst_pairs_train = 15
opath = os.path.join(out_dir, out_csv)
seed = 5
midinote = 60 # midi note to use
ticks_per_beat = 1000
velocity = 127
end_padding = 10
sustain = 1.0
dur = 1
subdiv = 1
num_trks = 2
beg_padding = 0
do_reverse = True

instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']

#instruments = ['Agogo','Woodblock']
inst_combos = [x for x in combinations(instruments, 2)]

um.shuffle_list(inst_combos)

poly_pairs = [(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(3,5),(5,7)]
train_poly = set([(3,5), (3,4), (4,5), (5,6), (6,7)])
valtest_poly = set([(2,3), (7,8), (5,7)])

bpm_bars = [(120, 2), (180, 3)]
runs = 2 if do_reverse == True else 1
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

fieldnames = ['name','inst1', 'inst2', 'bpm', 'num_bars', 'poly', 'pair', 'ratio', 'poly1', 'poly2', 'set']

bar_charts = ['bpm', 'poly', 'pair', 'set']
with open(opath, 'w') as csvf:
    csvw = csv.DictWriter(csvf,fieldnames=fieldnames)
    csvw.writeheader()
    for (cur_bpm, num_bars) in bpm_bars:
        for pair_idx,cur_pair in enumerate(inst_combos):
            # iterate over instruments
            ch_nums = [um.get_inst_program_number(x) for x in cur_pair]
            short_names = [''.join(x.split(' ')) for x in cur_pair]
            #print([x for x in zip(ch_nums, short_names)])
            
            for pnums in poly_pairs:
                # iterate over polyrhythm pairs

                for run in range(runs):
                    cur_set = 'train'
                    # run 1 = flip instruments
                    inst_order = [0,1] # instrument indices
                    if run == 1:
                        inst_order = [1,0]
                    # midi naming
                    inst1 = short_names[inst_order[0]]
                    inst2 = short_names[inst_order[1]]
                    short_pair = [inst1,inst2]
                    short_pair.sort()
                    pstr2 = '_'.join(short_pair)
                    if pair_idx >= inst_pairs_train and harder_split == False:
                        cur_set = um.coinflip_label(chance=0.5, label1='val', label2='test')

                    if harder_split == True and pnums in valtest_poly:
                        cur_set = um.coinflip_label(chance=0.5, label1='val', label2='test')
                    ratio = pnums[0]/pnums[1]
                    pstr = f"{pnums[0]}a{pnums[1]}"
                    outname = f"polyrhy-{inst1}_{inst2}-{cur_bpm}_{pstr}"
                    cur_row = {'inst1': inst1, 'inst2': inst2, 'poly': pstr, 'pair': pstr2, 'set': cur_set, 'bpm': cur_bpm, 'num_bars': num_bars,
                               'poly1': pnums[0], 'poly2': pnums[1], 'name': outname, 'ratio': ratio}
                    csvw.writerow(cur_row)
                    #print(outname)
                    # number of bars to do polyrhthm (polyrhythm isolated for one bar)
                    # do one instrument at a time



if data_profile == True:

    data = pl.scan_csv(opath).collect()
    for cat in bar_charts:
        um.profile_category(data, cat, ds=out_name, profile_type='overall')
    set_types = data['set'].value_counts()['set']
    for set_type in set_types:
        curset =  data.filter(pl.col('set') == set_type)
        for cat in bar_charts:
            if cat != 'set':
                um.profile_category(curset, cat, ds=out_name, profile_type=set_type, save_csv=data_csv, save_png = data_png)
                
        


