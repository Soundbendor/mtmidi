import util as um
import mido
import musicnoteconv as mnc
import numpy as np
import polyrhythms as PL

#max_num = 12
max_num=11
midinote = 60 # midi note to use
ticks_per_beat = 1000
cur_bpm = 120
velocity = 127
end_padding = 10
sustain = 1.0
dur = 1
subdiv = 1
num_trks = 2
beg_padding = 0
#num_offsets = 4 # not including no offset
#random_offsets = True
max_offset_ms = 250 # 16th note at 60 bpm
do_reverse = True


#instruments = ['Agogo','Woodblock']

# https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/dataset/music/midi.py#L21
runs = 2 if do_reverse == True else 1

#offset_ms_arr = [0.] + [((i+1.) * max_offset_ms)/num_offsets for i in range(num_offsets)]
#if random_offsets == True:
#offset_ms = [0] + [int(x) for x in um.get_random_list(1., float(max_offset_ms), num_offsets)]
# hand curated using [random.randint(1,251) for _ in range(4)]
#offset_ms_arr = [0, 67, 105, 170, 249]
# another hand curated random randint


for (cur_bpm, num_bars) in PL.bpm_bars:
    tempo_microsec = mido.bpm2tempo(cur_bpm)
    tick_offsets = {x:int(um.ms_to_ticks(x,ticks_per_beat = ticks_per_beat, bpm = cur_bpm)) for x in PL.offset_ms_arr}
    for offset_ms, offset_ticks in tick_offsets.items():
        for rvb_lvl, rvb_val in PL.reverb_lvl.items():

            for cur_pair in PL.inst_combos:
                # iterate over instruments
                ch_nums = [um.get_inst_program_number(x) for x in cur_pair]
                short_names = [''.join(x.split(' ')) for x in cur_pair]
                print([x for x in zip(ch_nums, short_names)])
                for pnums in PL.poly_pairs.keys():
                    # iterate over polyrhythm pairs
                    r1on, r1off = um.notedur_to_ticks(dur, subdiv = pnums[0], ticks_per_beat = ticks_per_beat, sustain = sustain)
                    r2on, r2off = um.notedur_to_ticks(dur, subdiv = pnums[1], ticks_per_beat = ticks_per_beat, sustain = sustain)
                    durs = [(r1on, r1off), (r2on, r2off)]
                    for run in range(runs):
                        mid = None
                        mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

                        # run 1 = flip instruments
                        inst_order = [0,1] # instrument indices
                        if run == 1:
                            inst_order = [1,0]
                        # midi naming
                        inst1 = short_names[inst_order[0]]
                        inst2 = short_names[inst_order[1]]
                        pstr = PL.get_pstr(pnums)
                        #outname = f"polyrhy-{inst1}_{inst2}-bpm_{cur_bpm}-rvb_{rvb_lvl}-offms_{offset_ms}-{pstr}.mid"
                        outname = PL.get_outname(inst1, inst2, cur_bpm, rvb_lvl, offset_ms, pstr)
                        #print(outname)
                        # number of bars to do polyrhthm (polyrhythm isolated for one bar)
                        # do one instrument at a time
                        for i in range(num_trks):
                            inst_idx = inst_order[i] # current instrument
                            _chnum = ch_nums[inst_idx]
                            mid.tracks.append(mido.MidiTrack())
                            mid.tracks[i].append(mido.Message('control_change', control=91, value=rvb_val, time =0, channel=i))
                            mid.tracks[i].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec, time = 0))
                            mid.tracks[i].append(mido.Message('program_change', program=_chnum, time =0, channel=i))
                            mid.tracks[i].name = short_names[inst_idx]


                        for i in range(num_trks):
                            pnum = pnums[i] # number of notes per other number of notes
                            #print(short_names[inst_idx])
                            d_on, d_off = durs[i] # current length of note_on and note_off
                            last_ticks_left = 0
                            for bar in range(num_bars):
                                # keep track of ticks left for rounding errors
                                # and make up for it at the start of next bar (or last note off)
                                ticks_left = (ticks_per_beat * 4)# 4/4 bar
                                # iterate over all notes to be played per bar
                                for cnum in range(pnum):
                                    first_beat = (bar == 0) and cnum == 0
                                    time_to_start = d_off
                                    # if first note, no need to delay note_on
                                    if cnum == 0:
                                        time_to_start = 0

                                    # pad beginning of midi file
                                    cur_padding = (beg_padding + offset_ticks) if first_beat == True else 0
                                        # delay by ticks left over from last bar but don't keep track of
                                    mid.tracks[i].append(mido.Message('note_on', note=midinote, velocity=velocity, time=time_to_start + last_ticks_left + cur_padding, channel=i))
                                    last_ticks_left = 0
                                    ticks_left -= time_to_start

                                    mid.tracks[i].append(mido.Message('note_off', note=midinote, velocity=velocity, time=d_on, channel=i))
                                    ticks_left -= d_on
                                    # if last note in bar, keep track of ticks left over
                                    if cnum == (pnum - 1):
                                        last_ticks_left = ticks_left
                            # after last bar, do the end_padding step (but remember to add ticks left over
                            if end_padding > 0 or last_ticks_left > 0:
                                if end_padding > 0:
                                    # only delay note_on by last ticks left, delay note_off by end_padding
                                    # so total is last ticks left + end_padding
                                    mid.tracks[i].append(mido.Message('note_on', note=midinote, velocity=0, time=last_ticks_left, channel=i))
                                    mid.tracks[i].append(mido.Message('note_off', note=midinote, velocity=0, time=end_padding, channel=i))
                                else:
                                    # don't delay note on, delay note off by last ticks left
                                    mid.tracks[i].append(mido.Message('note_on', note=midinote, velocity=0, time=0,channel=i))
                                    mid.tracks[i].append(mido.Message('note_off', note=midinote, velocity=0, time=end_padding, channel=i))
                            mid.tracks[i].append(mido.MetaMessage('end_of_track', time=0))
                        # end of run, save file
                        #mid.print_tracks()
                        um.save_midi(mid, outname)








