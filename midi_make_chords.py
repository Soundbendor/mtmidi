import util as um
import chords
import mido
import musicnoteconv as mnc

ticks_per_beat = 1000
cur_bpm = 120
velocity = 100
padding = 0
tempo_microsec = mido.bpm2tempo(cur_bpm)
sustain = 1.0
dur = 4
subdiv = 1



on_dur, off_dur = UM.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)

csvpath = os.path.join(UM.by_projpath('csv'), 'chords.csv')
outf = open(csvpath, 'w')
csvw = csv.DictWriter(outf,fieldnames=chords.fieldnames)
csvw.writeheader()



for cur_inst in UM.pitched_inst_to_use:
    short_inst = ''.join(cur_inst.split(' '))
    ch_num = UM.get_inst_program_number(cur_inst)
    for chord_type, cur_mnotes in chords.chord_midi.items():
        for offset_val, root_notename in chords.offsets:
            for inv_idx in range(chords.num_inversions):
                for rvb_lvl, rvb_val in UM.reverb_lvl.items():
                    inv_mnotes = chords.make_inversion(cur_mnotes, inv_idx)
                    offset_mnotes = chords.offset_notes(inv_mnotes, inv_idx)
                    tpose_mnotes, tposed_down = chords.transpose_to_range(offset_mnotes)
                    pitch = root_notename[:-1]
                    octave = 4
                    cur_root = root_notename
                    # since we are transposing up from c4
                    if tposed_down == True:
                        octave = 3
                        cur_root = pitch + '3'
                    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                    mid.tracks.append(mido.MidiTrack())
                    mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec))
                    mid.tracks[0].append(mido.Message('control_change', control=91, value=rvb_val, time =0, channel=chnum))
                    mid.tracks[0].append(mido.Message('program_change', program=ch_num, channel=0))
                    for i in range(num_notes):
                        cur_start = off_dur
                        cur_end = on_dur
                        if i == 0:
                            cur_delta = 0
                        for _mn in tpose_mnotes:
                            mid.tracks[0].append(mido.Message('note_on', note=_mn, velocity=velocity, time=cur_start, channel=0))
                        for _mn in tpose_mnotes:
                            mid.tracks[0].append(mido.Message('note_off', note=_mn, velocity=velocity, time=cur_end, channel=0))
                        """
                        if i == num_notes - 1:
                            mid.tracks[0].append(mido.Message('note_on', note=cur_mnotes[0], velocity = 0, time = off_dur, channel =0))
                            mid.tracks[0].append(mido.Message('note_off', note=cur_mnotes[0], velocity = 0, time = padding, channel=0))
                        """
                    mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
                    UM.save_midi(mid,outname)

outf.close()

