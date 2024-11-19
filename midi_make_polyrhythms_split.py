import numpy as np
import util as um
import musicnoteconv as mnc
from itertools import combinations
import csv
import random
import os
import polars as pl
from operator import itemgetter
import polyrhythms as PL

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


#instruments = ['Agogo','Woodblock']


#poly_pairs = [(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(3,5),(5,7)]
#train_poly = set([(3,5), (3,4), (4,5), (5,6), (6,7)])
#valtest_poly = set([(2,3), (7,8), (5,7)])

validtest_prop = int(.15 * PL.num_poly) # .15 for test, .15 for valid
first_prop = [x for (x,y) in PL.ptsort[:validtest_prop]] # first .15
last_prop = [x for (x,y) in PL.ptsort[-validtest_prop:]] # last .15
middle_prop = [x for (x,y) in PL.ptsort[validtest_prop:-validtest_prop]] # middle 0.7

train_poly = set(middle_prop)
valtest_poly = set(first_prop + last_prop) # combine first and last .15
#um.shuf_arr(firstlast_prop) # shuffle array (use util since set seed there)
#val_poly = set(firstlast_prop[:validtest_prop]) # validation gets first half of shuffled
#test_poly = set(firstlast_prop[validtest_prop:]) # testing gets second half of shuffled


#bpm_bars = [(120, 2), (180, 3)]
runs = 2 if do_reverse == True else 1
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

fieldnames = ['name','inst1', 'inst2', 'bpm', 'num_bars', 'poly', 'pair', 'ratio', 'norm_ratio', 'offset_ms', 'offset_ticks', 'rvb_lvl', 'rvb_val', 'poly1', 'poly2', 'set']

bar_charts = ['bpm', 'poly', 'pair', 'set']
with open(opath, 'w') as csvf:
    csvw = csv.DictWriter(csvf,fieldnames=fieldnames)
    csvw.writeheader()
    for (cur_bpm, num_bars) in PL.bpm_bars:
        #tempo_microsec = mido.bpm2tempo(cur_bpm)
        tick_offsets = {x:int(um.ms_to_ticks(x,ticks_per_beat = ticks_per_beat, bpm = cur_bpm)) for x in PL.offset_ms_arr}

        for offset_ms, offset_ticks in tick_offsets.items():
            for rvb_lvl, rvb_val in PL.reverb_lvl.items():
                for pair_idx,cur_pair in enumerate(PL.inst_combos):
                    # iterate over instruments
                    ch_nums = [um.get_inst_program_number(x) for x in cur_pair]
                    short_names = [''.join(x.split(' ')) for x in cur_pair]
                    #print([x for x in zip(ch_nums, short_names)])
                    
                    for pnums in PL.poly_pairs:
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
                            ratio = PL.get_ratio(pnums)
                            norm_ratio = PL.normalize_ratio(ratio)
                            pstr = PL.get_pstr(pnums)
                            outname = PL.get_outname(inst1, inst2, cur_bpm, rvb_lvl, offset_ms, pstr)
                            #outname = f"polyrhy-{inst1}_{inst2}-{cur_bpm}_{pstr}"
                            cur_row = {'inst1': inst1, 'inst2': inst2, 'poly': pstr, 'pair': pstr2, 'set': cur_set, 'bpm': cur_bpm, 'num_bars': num_bars,
                                       'rvb_lvl': rvb_lvl, 'rvb_val': rvb_val, 'offset_ms': offset_ms, 'offset_ticks': offset_ticks,
                                       'poly1': pnums[0], 'poly2': pnums[1], 'name': outname, 'ratio': ratio, 'norm_ratio': norm_ratio}
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
                
        


