import util as um
import mido
import musicnoteconv as mnc
from itertools import combinations



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
do_reverse = True

instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']

#instruments = ['Agogo','Woodblock']
inst_combos = [x for x in combinations(instruments, 2)]


poly_pairs = [(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(3,5),(5,7)]
bpm_bars = [(120, 2), (180, 3)]
runs = 2 if do_reverse == True else 1

for (cur_bpm, num_bars) in bpm_bars:
    tempo_microsec = mido.bpm2tempo(cur_bpm)
    for cur_pair in inst_combos:
        # iterate over instruments
        ch_nums = [um.get_inst_program_number(x) for x in cur_pair]
        short_names = [''.join(x.split(' ')) for x in cur_pair]
        print([x for x in zip(ch_nums, short_names)])
        for pnums in poly_pairs:
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
                pstr = f"{pnums[0]}a{pnums[1]}"
                outname = f"{inst1}_{inst2}-{cur_bpm}_{pstr}.mid"
                #print(outname)
                # number of bars to do polyrhthm (polyrhythm isolated for one bar)
                # do one instrument at a time
                for i in range(num_trks):
                    inst_idx = inst_order[i] # current instrument
                    _chnum = ch_nums[inst_idx]
                    mid.tracks.append(mido.MidiTrack())
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
                            time_to_start = d_off
                            # if first note, no need to delay note_on
                            if cnum == 0:
                                time_to_start = 0
                            # pad beginning of midi file
                            cur_padding = beg_padding if (bar == 0 and cnum == 0) else 0
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








