import util as UM
import mido
import musicnoteconv as mnc
import polars as pl
import chords as CH

# (x,x,5,4): V64-V53?
# ('min, 1,4,5,1) has minor 5



# gets orig_prog strings as defined above if is_modemix = False
# else gets mm_prog strings as defined above if is_modemix = True
def progtup_to_progstr(progtup, is_modemix=False):
    arr_str = []
    scaletype = progtup[0]
    if is_modemix == True:
        if scaletype == 'min':
            arr_str.append('mm2')
        else:
            arr_str.append('mm1')
    else:
        arr_str.append(scaletype)
    degstr = ''.join([str(deg) for deg in progtup[1:]])
    arr_str.append(degstr)
    retstr = '-'.join(arr_str)
    return retstr

#offsets are in sharp so looks weird but convert all flats to enharmonic sharps
progtypes = ['orig', 'mm']
# keycenter: pitch
# scale_type: maj, min, mm1 (maj to min), mm2 (picardy 3rd)
# orig_prog example: maj-1625 (modemix and original share same)
# mm_prog example: mm1-1625 (still the same as orig_prog for non-modemix progs
# (all scale degrees single digit so can merge with no separator)
modemix_fieldnames = ['name', 'inst', 'key_center', 'scale_type', 'is_modemix', 'orig_prog', 'mm_prog', 'bpm']
maj_diatonic = {1: ('c4', 'maj'),
                2: ('d4', 'min'),
                3: ('e4', 'min'),
                4: ('f4', 'maj'),
                5: ('g4', 'maj'),
                6: ('a4', 'min'),
                7: ('b4', 'dim')}

# abuse of "diatonic"
# just make g4 major
min_diatonic = {1: ('c4', 'min'),
                2: ('d4', 'dim'),
                3: ('ds4', 'maj'),
                4: ('f4', 'min'),
                5: ('g4', 'maj'),
                6: ('gs4', 'maj'),
                7: ('as4', 'maj')}

# just switch 4 to minor, 6 to flat6 major, 7 to flat7 major
# althuogh there isn't any flat7 to switch
modemix_diatonic = {1: ('c4', 'maj'),
                2: ('d4', 'min'),
                3: ('e4', 'min'),
                4: ('f4', 'min'),
                5: ('g4', 'maj'),
                6: ('gs4', 'maj'),
                7: ('b4', 'dim')}

chordprog_arr = [('maj', 1,4,5,1),
        ('maj', 1,4,6,5),
        ('maj', 1,5,6,4),
        ('maj', 1,6,4,5),
        ('maj', 2,5,1,6),
        ('maj', 4,1,5,6),
        ('maj', 5,4,1,5),
        ('maj', 5,6,4,1),
        ('maj', 6,4,1,5),
        #('min', 1,2,5,1),
        #('min', 1,3,4,1),
        #('min', 1,4,5,1),
        #('min', 1,6,7,1),
        #('min', 1,7,1,1),
        #('min', 6,7,6,1),
        ]

modemixprog_arr = [tuple([x]+list(y)) for y in chordprog_arr for x in progtypes]
mmp_to_idx = {i:x for (i,x) in enumerate(modemixrog_arr)}
idx_to_mmp = {x:i for (i,x) in mmp_to_idx.items()}
progtype_to_idx = {i:x for (i,x) in enumerate(progtypes)}
idx_to_progtype = {x:i for (i,x) in progtype_to_idx.items()}
# keyed by progtups (ie: ('maj', 1,4,5,1)) and has both original and modemix versions
# organized by 'orig' and 'mm' which have their own progressions (2-tuples with root,qual) and tup_str (the prog-specific tuple in string form)
chordprog_dict = {}

for progtup in chordprog_arr:
    scale_type = progtup[0]
    chord_dict = None
    orig_arr = []
    modemix_arr = []
    len_prog = len(progtup[1:])
    if 'maj' == scale_type:
        chord_dict = maj_diatonic
    else:
        chord_dict = min_diatonic
    for sdi, scale_deg in enumerate(progtup[1:]):
        orig_arr.append(chord_dict[scale_deg])
        # stuff for modemix prog
        if 'maj' == scale_type:
            modemix_arr.append(modemix_diatonic[scale_deg])
        else:
            if sdi == (len_prog - 1) and scale_deg == 1:
                # picardy third
                modemix_arr.append(maj_diatonic[scale_deg])
            else:
                # just normal minor chord prog
                modemix_arr.append(chord_dict[scale_deg])
    orig_str = progtup_to_progstr(progtup, is_modemix=False)
    mm_str = progtup_to_progstr(progtup, is_modemix=True)
    orig_prog_dict = {'prog': orig_arr, 'tup_str': orig_str}
    mm_prog_dict = {'prog': modemix_arr, 'tup_str': mm_str}
    chordprog_dict[progtup] = {'orig': orig_prog_dict, 'mm': mm_prog_dict, 'scale_type': scale_type}


def modemix_get_outname(progstr, inv_idx, short_inst, cur_root, ext = ""):
    ret = None
    outname = f'{short_inst}-{cur_root}-{progstr}_inv{inv_idx}'
    if len(ext) > 0:
        ret = f'{outname}.{ext}'
    else:
        ret = outname
    return ret


def debug():
    for pt in chordprog_arr:
        print("==========")
        print(pt)
        print("-----")
        print(chordprog_dict[pt]['orig']['prog'])
        print("-----")
        print(chordprog_dict[pt]['mm']['prog'])








