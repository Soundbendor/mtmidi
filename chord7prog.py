import util as UM
import mido
import musicnoteconv as mnc
import polars as pl
import chords as CH

# (x,x,5,4): V64-V53?
# ('min, 1,4,5,1) has minor 5



# e_{elt in seq changed}-degseq-s_{sub_type} 
# '' for base prog
# 'N' to specify no substitution
# 'S' to specify secondary dominant
# 'T' to specify tritone sub
def progtup_to_progstr(progtup, scale_type='', sub_type='N'):
    arr_str = []
    cur_elt = progtup[0]
    degstr = ''.join([str(deg) for deg in progtup[1:]])
    ret_str = ''
    if len(scale_type) > 0:
        ret_str = f'{scale_type}_'
    ret_str = ret_str + f'e_{cur_elt}-{degstr}'
    if len(sub_type) > 0:
        ret_str = ret_str + f'-s_{sub_type}'
    return ret_str

def get_dom_scale_deg(scaledeg):
    # for example 5 is the dominant/circle of fifths of 1
    ret = scaledeg + 4
    # 6 + 4 = 10 - 7 = 3
    while ret > 7:
        ret -= 7
    return ret

# shorthand for substitution types
sub_types = {'orig': 'N', 'secondary_dominant': 'S', 'tritone_sub': 'T'}

sub_type_arr = ['N', 'S', 'T']
sub_type_to_idx= {x:i for (i,x) in enumerate(sub_type_arr)}
idx_to_sub_type = {i:x for (x,i) in sub_type_to_idx.items()}


#offsets are in sharp so looks weird but convert all flats to enharmonic sharps

# keycenter: pitch
# scale_type: maj, min
# (all scale degrees single digit so can merge with no separator)
# sub_type = none, secondary_dominant, tritone_sub
# main_prog = base progression str (without the sub_type suffix)
# cur_prog = progression str with sub_type suffix
second_fieldnames = ['name', 'inst', 'key_center', 'scale_type', 'sub_type', 'base_prog', 'inv', 'sub_prog', 'bpm']
maj_diatonic = {1: ('c4', 'major7'),
                2: ('d4', 'minor7'),
                3: ('e4', 'minor7'),
                4: ('f4', 'major7'),
                5: ('g4', 'majorminor7'),
                6: ('a4', 'minor7'),
                7: ('b4', 'halfdim7')}

# abuse of "diatonic"
# just make g4 dom7
min_diatonic = {1: ('c4', 'minor7'),
                2: ('d4', 'halfdim7'),
                3: ('ds4', 'major7'),
                4: ('f4', 'minor7'),
                5: ('g4', 'majorminor7'),
                6: ('gs4', 'major7'),
                7: ('as4', 'majorminor7')}

# fifth_above in scale degrees
fifth_above = lambda x: ((x -1) % 7) + 1
prepend_tup = lambda x,y: tuple([x] + list(y))

# first element of tuple refers to which elt of seq to transform into a secondary dominant (1-indexed)
# applies to both major and minor
scale_type_arr = ['maj', 'min']
chordprog_arr = [(2, 1,6,2,5), (2, 1,1,4,5), (2,1,2,5,1), (2,1,3,6,5)]

subp_arr = []
for scale_type in scale_type_arr:
    for sub_type in sub_type_arr:
        for chordprog in chordprog_arr:
            cur_str = progtup_to_progstr(chordprog, scale_type=scale_type, sub_type=sub_type)
            subp_arr.append(cur_str)
subp_to_idx = {i:x for (i,x) in enumerate(subp_arr)}
idx_to_subp = {x:i for (i,x) in subp_to_idx.items()}

num_subprog = len(subp_arr)
num_subtypes = len(sub_type_arr)
# keyed by progtups (ie: ('maj', 1,4,5,1)) and has both original and modemix versions
# organized by 'orig' and 'mm' which have their own progressions (2-tuples with root,qual) and tup_str (the prog-specific tuple in string form)
chordprog_dict = {}

for progtup in chordprog_arr:
    change_elt = progtup[0]
    for scale_type in scale_type_arr:
        # to disambiguate, key chordprog_dict by adding 'maj' or 'min to beginning
        cur_key = prepend_tup(scale_type, progtup)
        chord_dict = None
        orig_arr = []
        secondary_arr = []
        tritonesub_arr = []
        if 'maj' == scale_type:
            chord_dict = maj_diatonic
        else:
            chord_dict = min_diatonic
        for idx in range(1, len(progtup)):
            scale_deg = progtup[idx] 
            deg_tup = chord_dict[scale_deg]
            orig_arr.append(deg_tup)
            # stuff for secondary/tritone prog
            if idx == change_elt:
                # won't specify sub for end of progtup anyways
                # so get the next scale deg
                next_scale_deg = progtup[idx+1]
                # want the note name
                next_deg_tup = chord_dict[next_scale_deg]
                next_note = next_deg_tup[0]
                # want the dominant of it (always a fifth above, even in minor)
                # but in octave 4
                dom_note = mnc.change_note_octave(mnc.transpose_note(next_note, 7), 4)
                secondary_arr.append((dom_note, 'majorminor7'))
                # tranpose note by tritone but keep within octave 4
                tritone_oct4 = mnc.change_note_octave(mnc.transpose_note(dom_note, 6), 4)
                tritonesub_arr.append((tritone_oct4, 'majorminor7'))
            else:
                secondary_arr.append(deg_tup)
                tritonesub_arr.append(deg_tup)

        main_str = progtup_to_progstr(progtup,  scale_type='', sub_type='')
        none_str = progtup_to_progstr(progtup,  scale_type=scale_type, sub_type='N')
        second_str = progtup_to_progstr(progtup, scale_type=scale_type, sub_type='S')
        tritone_str = progtup_to_progstr(progtup,  scale_type=scale_type, sub_type='T')
        orig_prog_dict = {'prog': orig_arr, 'tup_str': none_str}
        second_prog_dict = {'prog': secondary_arr, 'tup_str': second_str}
        tritone_prog_dict = {'prog': tritonesub_arr, 'tup_str': tritone_str}
        chordprog_dict[cur_key] = {'orig': orig_prog_dict, 'secondary_dominant': second_prog_dict, 'tritone_sub': tritone_prog_dict, 'scale_type': scale_type, 'tup_str': main_str}


# progstr should be the entire thing (with sub_type suffix)
def second_get_outname(progstr, inv_idx, short_inst, cur_root, ext = ""):
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
        for scale_type in ['maj','min']:
            cur_key = prepend_tup(scale_type, pt)
            print(scale_type, chordprog_dict[cur_key]['tup_str'])
            print("~~~~~")
            for cur_type in ['orig', 'secondary_dominant', 'tritone_sub']:
                print(cur_type + ":")
                print(chordprog_dict[cur_key][cur_type]['prog'])
                print(chordprog_dict[cur_key][cur_type]['tup_str'])
                print("-----")

#debug()







