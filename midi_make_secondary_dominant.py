import util as UM
import os, csv
import chords7 as CHS
import chord7prog as CSP
import mido
import musicnoteconv as mnc

ticks_per_beat = 2000
cur_bpm = 60
velocity = 100
padding = 30
tempo_microsec = mido.bpm2tempo(cur_bpm)
sustain = 0.95
dur = 4
subdiv = 1



on_dur, off_dur = UM.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)
#print(on_dur, off_dur)
csvpath = os.path.join(UM.by_projpath('csv'), 'secondary_dominant.csv')
outf = open(csvpath, 'w')
csvw = csv.DictWriter(outf,fieldnames=CSP.second_fieldnames)
csvw.writeheader()


#exit()
#for cur_inst in ['Baritone Sax']:
#for cur_inst in ['Bright Acoustic Piano']:
for cur_inst in UM.pitched_inst_to_use:
    short_inst = ''.join(cur_inst.split(' '))
    ch_num = UM.get_inst_program_number(cur_inst)
    for progtup, progdict in CSP.chordprog_dict.items():
        cur_scaletype = progdict['scale_type']
        base_progstr = progdict['tup_str']
        for progtype in ['orig', 'secondary_dominant', 'tritone_sub']:
            cur_sub_type = CSP.sub_types[progtype]
            cur_progstr = progdict[progtype]['tup_str']
            cur_prog = progdict[progtype]['prog']
            # key_offset is offseting the key center from c4:
            for key_offset, key_center in CHS.offsets.items():
                for inv_idx in range(CHS.num_inversions):
                    outname = CSP.second_get_outname(cur_progstr, inv_idx, short_inst, key_center, cur_scaletype, ext = "mid")
                    name = CSP.second_get_outname(cur_progstr, inv_idx, short_inst, key_center, cur_scaletype, ext = "")
                    cur_row = {'name': name, 'inst': short_inst, 'key_center': key_center, 'scale_type': cur_scaletype, 'sub_type': cur_sub_type, 'base_prog': base_progstr, 'sub_prog': cur_progstr, 'bpm': cur_bpm}
                    csvw.writerow(cur_row)
                    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                    mid.tracks.append(mido.MidiTrack())
                    mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec))
                    #mid.tracks[0].append(mido.Message('control_change', control=91, value=rvb_val, time =0, channel=0))
                    mid.tracks[0].append(mido.Message('program_change', program=ch_num, channel=0))

                    for chordtup in cur_prog:
                        cur_root = chordtup[0]
                        cur_qual = chordtup[1]
                        cur_mnotes = [mnc.note_to_midi(x) for x in CHS.chord_notes[cur_qual]]
                        prog_offset = CHS.note_offsets[cur_root] # offset from C in current chord progression
                        offset_val = key_offset + prog_offset
                        inv_mnotes = CHS.make_inversion(cur_mnotes, inv_idx)
                        offset_mnotes = CHS.offset_notes(inv_mnotes, offset_val)
                        tpose_mnotes, tposed_down = CHS.transpose_to_range(offset_mnotes)

                        for i in range(1):
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
                            if i == CHS.num_notes - 1:
                                mid.tracks[0].append(mido.Message('note_on', note=cur_mnotes[0], velocity = 0, time = off_dur, channel =0))
                                mid.tracks[0].append(mido.Message('note_off', note=cur_mnotes[0], velocity = 0, time = padding, channel=0))
                    mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))

                    UM.save_midi(mid, outname, dataset="secondary_dominant")

outf.close()

