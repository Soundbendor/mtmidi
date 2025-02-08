from itertools import combinations
from operator import itemgetter
import numpy as np
import polars as pl
from copy import deepcopy
from collections import OrderedDict

global ptsort, ptsort_norm, num_poly, smallest_ratio, biggest_ratio, ratio_gap, normalize_ratio
global polystr_to_idx, pair_to_str, rev_polystr_to_idx, class_arr, reg_polystr_to_idx, reg_pair_to_str, reg_rev_polystr_to_idx, reg_class_arr, default_tup

# for regression
# (0, 0) no match
default_tup = (0,0)


def get_pstr(pnums):
    pstr = f"{pnums[0]}a{pnums[1]}"
    return pstr

# let's prefix drumkit channel inst with P_

# channel 9, pitch 31 is Sticks
# channel 9, pitch 38 is Snare Drum
# channel 9, pitch 37 is Side Stick
train_prop = 0.7
instruments = ['Agogo','Woodblock','Taiko Drum','Melodic Tom','D:Snare Drum 1']
inst_combos = [x for x in combinations(instruments, 2)]
num_inst_combos = len(inst_combos)
inst_pairs_train = int(train_prop * num_inst_combos)
max_num = 12
max_range = max_num


poly_pairs = { (i,j): (i/j) for i in range(2,max_num+1) for j in range(2,min(i + max_range, max_num+1)) if (np.gcd(i,j) == 1 and i < j)}

# inst is shortname
fieldnames = ['name','inst1', 'inst2', 'bpm', 'num_bars', 'poly', 'pair', 'ratio', 'norm_ratio', 'offset_lvl', 'offset_ms', 'offset_ticks', 'rvb_lvl', 'rvb_val', 'poly1', 'poly2', 'polydist']

poly_tups = [((i,j),x) for (i,j),x in poly_pairs.items()]
ptsort = sorted(poly_tups, key=itemgetter(1))




# for classification

# for regression
# (0, 0) no match
default_tup = (0,0)
reg_poly_pairs_arr = [x for x in poly_pairs.keys()] + [default_tup] 
reg_polystr_to_idx = {get_pstr(x): i for i,x in enumerate(reg_poly_pairs_arr)} 
reg_pair_to_str = {x: get_pstr(x) for x in reg_poly_pairs_arr}
reg_rev_polystr_to_idx = {i:x for (x,i) in reg_polystr_to_idx.items()}
reg_class_arr = [k for (k,v) in reg_polystr_to_idx.items()]



# 0 is baseline
# 1,2, witin 16th note at 60 bpm (250)
# 3,4 within 250 and 0.4 * 4000 = 1600 of sample (because slowest polyrhythm is 2:someting)
# first run for 3,4 = array([ 700.23030532, 1067.37954942])
offset_ms_arr = [0, 120, 204, 700, 1067]
bpm_bars = [(60, 1),(120, 2), (180, 3)]

def get_nearest_poly(normed_pred, thresh=0.001, as_str = True):
    _start_idx = 0
    _end_idx = len(ptsort_norm)
    def _get_nearest(start_idx, end_idx, ipt):
        mid_idx = (end_idx+start_idx)//2
        mid_val = ptsort_norm[mid_idx][1]
        match = np.isclose(ipt, mid_val, atol=thresh)
        #print(f"idx0: {start_idx}, idx1:{end_idx}, ipt:{ipt}, mididx:{mid_idx}, midval:{mid_val}, match:{match}")
        _ret = default_tup
        if match == True:
            _ret = ptsort_norm[mid_idx][0]
        else:
            if ipt < mid_val:
                start = start_idx
                end = mid_idx
                if end >start:
                    #print(f'recurse left, start:{start}, end:{end}')
                    _ret = _get_nearest(start,end, ipt)
            else:
                start = mid_idx+1
                end = end_idx
                if end >start:
                    #print(f'recurse right, start:{start}, end:{end}')
                    _ret = _get_nearest(start,end, ipt)
        return _ret
    ret = _get_nearest(_start_idx, _end_idx, normed_pred)
    if as_str == True:
        ret = reg_pair_to_str[ret]
    return ret

def get_idx_from_polystr(_pstr):
    return polystr_to_idx[_pstr]

def get_rev_idx_from_polystr(_pstr):
    return rev_polystr_to_idx[_pstr]


def get_ratio(pnums):
    ratio = float(pnums[0])/pnums[1]
    return ratio

def get_outname(inst1, inst2, cur_bpm, rvb_lvl, offset_ms, pstr, with_ext= True):
    outname = f"{inst1}_{inst2}-bpm_{cur_bpm}-rvb_{rvb_lvl}-offms_{offset_ms}-{pstr}"
    if with_ext == True:
        outname = f"{outname}.mid"
    return outname

def init(poly_df, is_classification):

    global ptsort, ptsort_norm, num_poly, smallest_ratio, biggest_ratio, ratio_gap, normalize_ratio
    global polystr_to_idx, pair_to_str, rev_polystr_to_idx, class_arr, reg_polystr_to_idx, reg_pair_to_str, reg_rev_polystr_to_idx, reg_class_arr
    
   
    ptsort_keys = sorted([tuple([int(y) for y in x.split('a')]) for x in poly_df['poly'].unique()], key=get_ratio)
    ptsort_ratios = sorted([get_ratio(x) for x in ptsort_keys])
    
    smallest_ratio = ptsort_ratios[0]
    biggest_ratio = ptsort_ratios[-1]
    
    ratio_gap = biggest_ratio - smallest_ratio
    normalize_ratio = lambda x: (x - smallest_ratio)/ratio_gap # set ratios to be between 0 and 1
    
    ptsort_norm_ratios = [normalize_ratio(x) for x in ptsort_ratios]
    ptsort = [(x,y) for (x,y) in zip(ptsort_keys, ptsort_ratios)]
    ptsort_norm = [(x,y) for (x,y) in zip(ptsort_keys, ptsort_norm_ratios)]

    num_poly = len(ptsort_keys)
    poly_pairs = {x:y for (x,y) in zip(ptsort_keys, ptsort_ratios)}

    # classification
    polystr_to_idx = {get_pstr(x): i for i,x in enumerate(ptsort_keys)} 
    pair_to_str = {x: get_pstr(x) for x in ptsort_keys}
    rev_polystr_to_idx = {i:x for (x,i) in polystr_to_idx.items()}
    class_arr = [k for (k,v) in polystr_to_idx.items()]

    default_tup = (0,0)
    reg_poly_pairs_arr = deepcopy(ptsort_keys) + [default_tup] 
    reg_polystr_to_idx = {get_pstr(x): i for i,x in enumerate(reg_poly_pairs_arr)} 
    reg_pair_to_str = {x: get_pstr(x) for x in reg_poly_pairs_arr}
    reg_rev_polystr_to_idx = {i:x for (x,i) in reg_polystr_to_idx.items()}
    reg_class_arr = [k for (k,v) in reg_polystr_to_idx.items()]

    ret = None
    if is_classification == True: 
        ret = poly_df.with_columns(label_idx=pl.col('poly').replace_strict(polystr_to_idx).cast(int))
    else:
        ret = poly_df.with_columns(label_idx=pl.col('poly').replace_strict(reg_polystr_to_idx).cast(int))
    return ret
