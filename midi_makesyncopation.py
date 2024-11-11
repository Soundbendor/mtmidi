import util as um
import mido
import musicnoteconv as mnc
from itertools import combinations

# copied over from midi_makedynamics
dyn_ltr = ["pp", "p", "mp", "mf", "f", "ff"]
# (127 - 22)/(num - 1)
dyn_num = [22, 43, 64, 85, 106, 127]
dyn = {x:y for (x,y) in zip(dyn_ltr, dyn_num)}




midinote = 60 # midi note to use
ticks_per_beat = 1000
cur_bpm = 120
end_padding = 10
tempo_microsec = mido.bpm2tempo(cur_bpm)
sustain = 1.0
beg_padding = 0
num_bars = 2
num_beats = 4

accent_dyn = 'ff'
reg_dyn = 'mp'
acc_num = dyn[accent_dyn]
reg_num = dyn[reg_dyn]

instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']

#instruments = ['Agogo','Woodblock']

# 1-bar patterns repeated over num_bars

# naming scheme inst-{dur x subdiv}_{pattern}_{beat}.mid
# where pattern refers to where the accent lies in the single different beat (0-indexed)
# so pattern goes 0,1,2,...,(subdiv-1)
# and beat is where this different-accent beat lies in the measure (0-indexed)
# so beat goes 0,1,2,3

# note that 0 is the "normal" placement of the accent so there should only be one entry for {x}_0_0
# (and call it that)

dur_subdiv = [(4,4),(4,3)]

for (dur,subdiv) in dur_subdiv:
    d_on, d_off = um.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)
    notes_per_bar = dur * subdiv

    for _inst in instruments:
        ch_num = um.get_inst_program_number(_inst)
        short_name = ''.join(_inst.split(' '))
        for patt in range(subdiv):
            beat_cycles = num_beats
            # accent on downbeat, only need to make one of these
            if patt == 0:
                beat_cycles = 1
            for beat_cycle in range(beat_cycles):
                # now this is the midi outer loop
                mid = None
                mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                mid.tracks.append(mido.MidiTrack())
                mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec, time = 0))
                mid.tracks[0].append(mido.Message('program_change', program=ch_num, time =0, channel=0))
                outname = f"{short_name}-{notes_per_bar}_{patt}_{beat_cycle}.mid"
                # unlike other scripts, try to align by beat
                # keep track of ticks left from last beat
                last_ticks_left = 0

                for cur_bar in range(num_bars):
                    for beat in range(num_beats):
                        # keep track of ticks left over
                        ticks_left = ticks_per_beat
                        dyn_patt = [reg_num] * subdiv # default dynamic pattern for beat as the "regular" dynamic
                        if patt == 0 or beat != beat_cycle:
                            dyn_patt[0] = acc_num #downbeat accent
                        else:
                            dyn_patt[patt] = acc_num #accent corresponds to patt index
                        for _sd in range(subdiv):
                            _v = dyn_patt[_sd]
                            start_time = d_off
                            if _sd == 0:
                                start_time = 0
                            mid.tracks[0].append(mido.Message('note_on', note=midinote, velocity=_v, time=start_time + last_ticks_left, channel=0))
                            ticks_left -= start_time
                            last_ticks_left = 0
                            end_time = d_on
                            mid.tracks[0].append(mido.Message('note_off', note=midinote, velocity=_v, time=end_time, channel=0))
                            ticks_left -= end_time
                            if _sd == (subdiv - 1):
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

                               




