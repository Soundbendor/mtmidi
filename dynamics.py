import musicnoteconv as mnc
from itertools import combinations
import mido

inflection_marker = "-" # marks dynamic types with inflection

instruments = ['Agogo','Woodblock','Taiko Drum','Melodic Tom','D:Snare Drum 1']

# inst is shortname
fieldnames = ['name','dyn1', 'dyn2', 'dyn_pair', 'inst', 'inflection_point', 'dyn_category', 'dyn_subcategory',
              'rvb_lvl', 'rvb_val', 'offset_lvl', 'offset_ms', 'offset_ticks', 
              'beats_per_bar', 'num_bars', 'beat_subdiv', 'bpm', 'num_beats']

dyn_ltr = ["pp", "p", "mp", "mf", "f", "ff"]
# (127 - 22)/(num - 1)
dyn_num = [22, 43, 64, 85, 106, 127]
dyn = {x:y for (x,y) in zip(dyn_ltr, dyn_num)}

# note: doesn't go all the way to end_dyn (one short)
def make_dynamic_ramp(start_num, end_num, num_notes):
    return [int(start_num + i*((end_num - start_num)/num_notes)) for i in range(num_notes)]

def make_dyn_pair_str(_dyn1, _dyn2):
    return "-".join([_dyn1, _dyn2])

# flat = stay at first dynamic of pair
# hairpin = go from dyn1 to dyn2 to dyn1 smoothly over (div1 + div2)
#revhairpin = go from dyn2 to dyn1 to dyn2 smoothyl over (div1 + div2)
# cresc = go from dyn1 to dyn2 smoothly over 1
# decresc = go from dyn2 to dyn1 smoothyl over 1
# subf = go from dyn1 to dyn1 (dyn1 over div1,dyn2 over div2)
# subp = go from dyn2 to dyn1 (dyn2 over div1, dyn1 over div2)
# dyn1 < dyn2
def get_velocities(_dyn1, _dyn2, beat_subdiv = (4,2), start_beat_dyn2 = 1, dyn_category="flat"):
    notes_per_beat = beat_subdiv[1]
    num_beats = beat_subdiv[0]
    total_num_notes = num_beats * notes_per_beat
    num_notes = (int(notes_per_beat * start_beat_dyn2), int(notes_per_beat * (num_beats - start_beat_dyn2)))
    start_dyn_num = dyn[_dyn1]
    end_dyn_num = dyn[_dyn2]
    # num_beats should be even...
    ret = []
    if dyn_category == "flat":
        ret = [start_dyn_num] * total_num_notes
    elif dyn_category == "hairpin":
        ret1 = make_dynamic_ramp(start_dyn_num, end_dyn_num, num_notes[0])
        ret2 = make_dynamic_ramp(end_dyn_num, start_dyn_num, num_notes[1])
        ret = ret1 + ret2
    elif dyn_category == "revhairpin":
        ret1 = make_dynamic_ramp(end_dyn_num, start_dyn_num, num_notes[0])
        ret2 = make_dynamic_ramp(start_dyn_num, end_dyn_num, num_notes[1])
        ret = ret1 + ret2
    elif dyn_category == "cresc":
        ret = make_dynamic_ramp(start_dyn_num, end_dyn_num, total_num_notes)
    elif dyn_category == "decresc":
        ret = make_dynamic_ramp(end_dyn_num, start_dyn_num, total_num_notes)
    elif dyn_category == "subf":
        ret1 = [start_dyn_num] * num_notes[0]
        ret2 = [end_dyn_num] * num_notes[1]
        ret = ret1 + ret2
    elif dyn_category == "subp":
        ret1 = [end_dyn_num] * num_notes[0]
        ret2 = [start_dyn_num] * num_notes[1]
        ret = ret1 + ret2
    return ret



num_bars = 1
bpm = 60

# curated from array([ 73.38251409, 263.56219544, 327.15342688, 420.31061945, 430.5955094 ])
offset_ms_arr = [0, 264, 431]

dyn1 = ["pp", "p", "mp"]
dyn2 = ["mf", "f", "ff"]
dyn_pairs = [(x,y) for x in dyn1 for y in dyn2]
        
# total number of notes = beat * subdiv
# tuples of number of beats and subdivisions
beat_subdiv_arr = [(4,2),(4,3), (4,4),(4,5),(4,6),(4,7),(4,8)]

# one dynamic
dyn_cat_one = ["flat"]

# two dynamics but from beginning to end
dyn_cat_two = ["cresc", "decresc"]

dyn_cat_three = ['hairpin', 'revhairpin', 'subp', 'subf']
# two dynamics but can change dynamic inflection point
# name formatL {type}-{start_of_dyn2}
dyn_subcat_three = [f"{x}{inflection_marker}{y}" for x in dyn_cat_three for y in range(1,4)]

dyn_categories = dyn_cat_one + dyn_cat_two + dyn_cat_three
dyn_subcategories = dyn_cat_one + dyn_cat_two + dyn_subcat_three

num_categories = len(dyn_categories)
num_subcategories = len(dyn_subcategories)

dyn_category_to_idx = {x:i for (i,x) in enumerate(dyn_categories)}
dyn_idx_to_category = {i:x for (x,i) in dyn_category_to_idx.items()}
dyn_subcategory_to_idx = {x:i for (i,x) in enumerate(dyn_subcategories)}
dyn_idx_to_subcategory = {i:x for (x,i) in dyn_subcategory_to_idx.items()}

def get_category_idx(category):
    return dyn_category_to_idx[category]

def get_subcategory_idx(subcategory):
    return dyn_subcategory_to_idx[subcategory]

def has_inflection_pt(_dyn):
    return inflection_marker in _dyn

def get_dyn_category_and_inflection_pt(_dyn):
    dsplit = _dyn.split("-")
    return dsplit[0], int(dsplit[1])

def get_outname(_dyn_subcat, _dyn1, _dyn2, _inst_short, _nbeats, _beat_sd, _rvbval, _offms, ext=""):
    ret = None
    base_str = f"{_dyn_subcat}_{_dyn1}-{_dyn2}_{_inst_short}_beat{_nbeats}-{_beat_sd}_rvb{_rvbval}_off{_offms}"
    if len(ext) > 0:
        ret = f"{base_str}.{ext}"
    else:
        ret = base_str
    return ret

