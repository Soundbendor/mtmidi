import util as um
import mido
import musicnoteconv as mnc
from itertools import combinations

dyn_ltr = ["pp", "p", "mp", "mf", "f", "ff"]
# (127 - 22)/(num - 1)
dyn_num = [22, 43, 64, 85, 106, 127]
dyn = {x:y for (x,y) in zip(dyn_ltr, dyn_num)}

# note: doesn't go all the way to end_dyn (one short)
def make_dynamic_ramp(start_num, end_num, num_notes):
    return [int(start_num + i*((end_num - start_num)/num_notes)) for i in range(num_notes)]


# flat = stay at first dynamic of pair
# hairpin = go from dyn1 to dyn2 to dyn1 smoothly over (1/2 + 1/2)
#revhairpin = go from dyn2 to dyn1 to dyn2 smoothyl over (1/2 + 1/2)
# cresc = go from dyn1 to dyn2 smoothly over 1
# decresc = go from dyn2 to dyn1 smoothyl over 1
# subf = go from dyn1 to dyn1 (1/2 at dyn1, 1/2 at dyn2)
# subp = go from dyn2 to dyn1 (1/2 at dyn2, 1/2 at dyn1)
# dyn1 < dyn2
def get_velocities(dyn1, dyn2, num_notes, dyn_type="flat"):
    start_num = dyn[dyn1]
    end_num = dyn[dyn2]
    # num_notes should be even...
    half_num = int(num_notes/2)
    ret = []
    if dyn_type == "flat":
        ret = [start_num] * num_notes
    elif dyn_type == "hairpin":
        ret1 = make_dynamic_ramp(start_num, end_num, half_num)
        ret2 = make_dynamic_ramp(end_num, start_num, half_num)
        ret = ret1 + ret2
    elif dyn_type == "revhairpin":
        ret1 = make_dynamic_ramp(end_num, start_num, half_num)
        ret2 = make_dynamic_ramp(start_num, end_num, half_num)
        ret = ret1 + ret2
    elif dyn_type == "cresc":
        ret = make_dynamic_ramp(start_num, end_num, num_notes)
    elif dyn_type == "decresc":
        ret = make_dynamic_ramp(end_num, start_num, num_notes)
    elif dyn_type == "subf":
        ret1 = [start_num] * half_num
        ret2 = [end_num] * half_num
        ret = ret1 + ret2
    elif dyn_type == "subp":
        ret1 = [end_num] * half_num
        ret2 = [start_num] * half_num
        ret = ret1 + ret2
    return ret



midinote = 60 # midi note to use
ticks_per_beat = 1000
cur_bpm = 120
end_padding = 10
tempo_microsec = mido.bpm2tempo(cur_bpm)
sustain = 1.0
beg_padding = 0
num_bars = 2

instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']

#instruments = ['Agogo','Woodblock']

# always goes from soft to loud
#dyn_to_use = dyn_ltr
dyn_to_use = ["p", "ff"]
dyn_pair = [x for x in combinations(dyn_to_use, 2)]
# total number of notes = dur * subdiv * num_bars
dur_subdiv = [(4,2),(4,3)]

dyn_type = ["flat", "hairpin", "revhairpin", "cresc", "decresc", "subf", "subp"]

for _ds in dur_subdiv:
    dur = _ds[0]
    subdiv = _ds[1]
    notes_per_bar = dur * subdiv
    total_notes = notes_per_bar * num_bars
    for _dt in dyn_type:
        for _dp in dyn_pair:
            dyn1 = _dp[0]
            dyn2 = _dp[1]
            _v = get_velocities(dyn1, dyn2, total_notes, dyn_type=_dt)
            for _inst in instruments:
                ch_num = um.get_inst_program_number(_inst)
                short_name = ''.join(_inst.split(' '))
                mid = None
                mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                mid.tracks.append(mido.MidiTrack())
                mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec, time = 0))
                mid.tracks[0].append(mido.Message('program_change', program=ch_num, time =0, channel=0))
                d_on, d_off = um.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)
                dynstr = f"{_dt}-{dyn1}_{dyn2}"
                durstr = f"{dur}_{subdiv}"
                outname = f"{short_name}-{dynstr}-{durstr}.mid"
                #print(outname)
                # keep track of ticks left from last bar
                last_ticks_left = 0
                for curbar in range(num_bars):
                    # keep track of ticks left over
                    ticks_left = ticks_per_beat * 4 # 4/4 bar
                    for notenum in range(notes_per_bar):
                        start_time = d_off
                        realnum = notenum + (curbar * notes_per_bar)
                        curvel = _v[realnum]
                        if notenum == 0:
                            start_time = 0
                        
                        mid.tracks[0].append(mido.Message('note_on', note=midinote, velocity=curvel, time=start_time + last_ticks_left, channel=0))
                        ticks_left -= start_time
                        last_ticks_left = 0
                        end_time = d_on
                        mid.tracks[0].append(mido.Message('note_off', note=midinote, velocity=curvel, time=end_time, channel=0))
                        ticks_left -= end_time
                        if notenum == (notes_per_bar) - 1:
                            last_ticks_left = max(0,ticks_left)
                if end_padding > 0 or last_ticks_left > 0:
                    if end_padding > 0:
                        # only delay note_on by last ticks left, delay note_off by end_padding
                        # so total is last ticks left + end_padding
                        mid.tracks[0].append(mido.Message('note_on', note=midinote, velocity=0, time=last_ticks_left, channel=0))
                        mid.tracks[0].append(mido.Message('note_off', note=midinote, velocity=0, time=end_padding, channel=0))
                    else:
                        # don't delay note on, delay note off by last ticks left
                        mid.tracks[0].append(mido.Message('note_on', note=midinote, velocity=0, time=0,channel=0))
                        mid.tracks[0].append(mido.Message('note_off', note=midinote, velocity=0, time=end_padding, channel=0))
                mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
                um.save_midi(mid, outname)




                
