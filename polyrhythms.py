from itertools import combinations
from operator import itemgetter
import numpy as np

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
max_num = 11
max_range = 6
poly_pairs = { (i,j): (i/j) for i in range(2,max_num+1) for j in range(2,min(i + max_range, max_num+1)) if (np.gcd(i,j) == 1 and i < j)}



poly_tups = [((i,j),x) for (i,j),x in poly_pairs.items()]
ptsort = sorted(poly_tups, key=itemgetter(1))

num_poly = len(ptsort)

smallest_ratio = ptsort[0][1]
biggest_ratio = ptsort[-1][1]
ratio_gap = biggest_ratio - smallest_ratio
normalize_ratio = lambda x: (x - smallest_ratio)/ratio_gap # set ratios to be between 0 and 1

poly_tups_norm = [((i,j),normalize_ratio(x)) for (i,j),x in ptsort]


# for classification
poly_pairs_arr = [x for x in poly_pairs.keys()]
polystr_to_idx = {get_pstr(x): i for i,x in enumerate(poly_pairs_arr)} 
pair_to_str = {x: get_pstr(x) for x in poly_pairs_arr}
rev_polystr_to_idx = {i:x for (x,i) in polystr_to_idx.items()}
class_arr = [k for (k,v) in polystr_to_idx.items()]

# for regression
# (0, 0) no match
default_tup = (0,0)
reg_poly_pairs_arr = [x for x in poly_pairs.keys()] + [default_tup] 
reg_polystr_to_idx = {get_pstr(x): i for i,x in enumerate(reg_poly_pairs_arr)} 
reg_pair_to_str = {x: get_pstr(x) for x in reg_poly_pairs_arr}
reg_rev_polystr_to_idx = {i:x for (x,i) in reg_polystr_to_idx.items()}
reg_class_arr = [k for (k,v) in reg_polystr_to_idx.items()]




offset_ms_arr = [0, 120, 204]
bpm_bars = [(60, 1),(120, 2), (180, 3)]
reverb_lvl = {0:0, 1: 63, 2:127}

def get_nearest_poly(normed_pred, thresh=0.001, as_str = True):
    _start_idx = 0
    _end_idx = len(poly_tups_norm)
    def _get_nearest(start_idx, end_idx, ipt):
        mid_idx = (end_idx+start_idx)//2
        mid_val = poly_tups_norm[mid_idx][1]
        match = np.isclose(ipt, mid_val, atol=thresh)
        #print(f"idx0: {start_idx}, idx1:{end_idx}, ipt:{ipt}, mididx:{mid_idx}, midval:{mid_val}, match:{match}")
        _ret = default_tup
        if match == True:
            _ret = poly_tups_norm[mid_idx][0]
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



def get_ratio(pnums):
    ratio = pnums[0]/pnums[1]
    return ratio

def get_outname(inst1, inst2, cur_bpm, rvb_lvl, offset_ms, pstr, with_ext= True):
    outname = f"polyrhy-{inst1}_{inst2}-bpm_{cur_bpm}-rvb_{rvb_lvl}-offms_{offset_ms}-{pstr}"
    if with_ext == True:
        outname = f"{outname}.mid"
    return outname
