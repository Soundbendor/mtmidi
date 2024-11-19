from itertools import combinations
from operator import itemgetter
import numpy as np

instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']
inst_combos = [x for x in combinations(instruments, 2)]
max_num = 11
poly_pairs = { (i,j): (i/j) for i in range(2,max_num+1) for j in range(2,max_num+1) if (np.gcd(i,j) == 1 and i < j)}
polydict = {f"{n1}a{n2}": i for i,(n1,n2) in enumerate(poly_pairs.keys())} 
poly_tups = [((i,j),x) for (i,j),x in poly_pairs.items()]
ptsort = sorted(poly_tups, key=itemgetter(1))

rev_polydict = {i:x for (x,i) in polydict.items()}
class_arr = [k for (k,v) in polydict.items()]

smallest_ratio = ptsort[0][1]
biggest_ratio = ptsort[-1][1]
ratio_gap = biggest_ratio - smallest_ratio
normalize_ratio = lambda x: (x - smallest_ratio)/ratio_gap # set ratios to be between 0 and 1

num_poly = len(ptsort)

offset_ms_arr = [0, 120, 204]
bpm_bars = [(60, 1),(120, 2), (180, 3)]
reverb_lvl = {0:0, 1: 63, 2:127}

def get_pstr(pnums):
    pstr = f"{pnums[0]}a{pnums[1]}"
    return pstr

def get_ratio(pnums):
    ratio = pnums[0]/pnums[1]
    return ratio

def get_outname(inst1, inst2, cur_bpm, rvb_lvl, offset_ms, pstr):
    outname = f"polyrhy-{inst1}_{inst2}-bpm_{cur_bpm}-rvb_{rvb_lvl}-offms_{offset_ms}-{pstr}.mid"
    return outname
