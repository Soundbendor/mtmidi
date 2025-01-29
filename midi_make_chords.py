import util as UM
import os, csv
import chords
import mido
import musicnoteconv as mnc

ticks_per_beat = 1000
cur_bpm = 60
velocity = 100
padding = 10
tempo_microsec = mido.bpm2tempo(cur_bpm)
sustain = 0.9
dur = 4
subdiv = 1



on_dur, off_dur = UM.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)
#print(on_dur, off_dur)
csvpath = os.path.join(UM.by_projpath('csv'), 'chords.csv')
outf = open(csvpath, 'w')
csvw = csv.DictWriter(outf,fieldnames=chords.fieldnames)
csvw.writeheader()


#exit()
for cur_inst in UM.pitched_inst_to_use:
#for cur_inst in ['Baritone Sax']:
    short_inst = ''.join(cur_inst.split(' '))
    ch_num = UM.get_inst_program_number(cur_inst)
    for chord_quality, cur_mnotes in chords.chord_midi.items():
        for offset_val, root_notename in chords.offsets.items():
            for inv_idx in range(chords.num_inversions):
                inv_mnotes = chords.make_inversion(cur_mnotes, inv_idx)
                offset_mnotes = chords.offset_notes(inv_mnotes, offset_val)
                tpose_mnotes, tposed_down = chords.transpose_to_range(offset_mnotes)
                pitch = root_notename[:-1]
                octave = 4
                cur_root = root_notename
                # since we are transposing up from c4
                if tposed_down == True:
                    octave = 3
                    cur_root = pitch + '3'
                outname = chords.get_outname(chord_quality, inv_idx, short_inst, cur_root, ext = "mid")
                name = chords.get_outname(chord_quality, inv_idx, short_inst, cur_root, ext = "")
                cur_row = {'name': name, 'root': cur_root, 'pitch': pitch, 'octave': octave, 'quality': chord_quality, 'inversion': inv_idx, 'inst': short_inst, 'quality_idx': chords.quality_to_idx[chord_quality], 'bpm': cur_bpm}
                csvw.writerow(cur_row)
                mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                mid.tracks.append(mido.MidiTrack())
                mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec))
                #mid.tracks[0].append(mido.Message('control_change', control=91, value=rvb_val, time =0, channel=0))
                mid.tracks[0].append(mido.Message('program_change', program=ch_num, channel=0))
                for i in range(chords.num_notes):
                    cur_start = off_dur
                    cur_end = on_dur
                    if i == 0:
                        cur_start = 0
                    for midx, _mn in enumerate(tpose_mnotes):
                        note_start = cur_start if midx == 0 else 0 
                        mid.tracks[0].append(mido.Message('note_on', note=_mn, velocity=velocity, time=note_start, channel=0))
                    for midx, _mn in enumerate(tpose_mnotes):
                        note_end = cur_end if midx == 0 else 0 
                        mid.tracks[0].append(mido.Message('note_off', note=_mn, velocity=velocity, time=note_end, channel=0))
                    if i == chords.num_notes - 1:
                        mid.tracks[0].append(mido.Message('note_on', note=cur_mnotes[0], velocity = 0, time = off_dur, channel =0))
                        mid.tracks[0].append(mido.Message('note_off', note=cur_mnotes[0], velocity = 0, time = padding, channel=0))
                mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))

                UM.save_midi(mid, outname, dataset="chords")

outf.close()

