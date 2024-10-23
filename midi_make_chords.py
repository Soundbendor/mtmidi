import util as um
import mido
import musicnoteconv as mnc

def make_inversion(cur_midinotes, inv):
    if inv == 0:
        return cur_midinotes
    else:
        notes = [x for x in cur_midinotes[inv:]]
        notes2 = [x + 12 for x in cur_midinotes[:inv]]
        ret = notes + notes2
        return ret



num_inversions = 4
ticks_per_beat = 1000
cur_bpm = 120
velocity = 100
padding = 10
tempo_microsec = mido.bpm2tempo(cur_bpm)
num_notes = 4
sustain = 1.0
dur = 4
subdiv = 1
offsets = {0: 'c4', 1: 'cs4', 2: 'd4', 3: 'ds4', 4: 'e4',
          5: 'f4', 6: 'fs4', 7: 'g4', 8: 'gs4', 9: 'a4',
          10: 'as4', 11: 'b4'}
chord_notes = {'major7': ['c4', 'e4', 'g4', 'b4'],
          'minor7': ['c4', 'ef4', 'g4', 'bf4'],
          'majorminor7': ['c4', 'e4', 'g4', 'bf4'],
          'minormajor7': ['c4', 'ef4', 'g4', 'b4'],
          'augmajor7': ['c4', 'e4', 'gs4', 'b4'],
          'halfdim7': ['c4', 'ef4', 'gf4', 'bf4'],
          'fulldim7': ['c4', 'ef4', 'gf4', 'a4']
          }
num_offsets = 0

chord_midi = {}
for chord_type, chord_arr in chord_notes.items():
    chord_midi[chord_type] = [mnc.note_to_midi(x) for x in chord_arr]

instruments = ['Acoustic Grand Piano']
on_dur, off_dur = um.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)


for cur_inst in instruments:
    short_inst = ''.join(cur_inst.split(' '))
    ch_num = um.get_inst_program_number(cur_inst)
    for chord_type, cur_mnotes in chord_midi.items():
        for cur_offset in range(num_offsets+1):
            for inv_idx in range(num_inversions):
                inv_mnotes = make_inversion(cur_mnotes, inv_idx) 
                mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
                mid.tracks.append(mido.MidiTrack())
                mid.tracks[0].append(mido.MetaMessage('set_tempo', tempo = tempo_microsec))
                mid.tracks[0].append(mido.Message('program_change', program=ch_num, channel=0))
                offset_name = offsets[cur_offset]
                outname = f'{short_inst}-{chord_type}_inv{inv_idx}_{offset_name}.mid'
                for i in range(num_notes):
                    cur_start = off_dur
                    cur_end = on_dur
                    if i == 0:
                        cur_delta = 0
                    for _mn in inv_mnotes:
                        cur_mn = _mn + cur_offset
                        mid.tracks[0].append(mido.Message('note_on', note=cur_mn, velocity=velocity, time=cur_start, channel=0))
                    for _mn in inv_mnotes:
                        cur_mn = _mn + cur_offset
                        mid.tracks[0].append(mido.Message('note_off', note=cur_mn, velocity=velocity, time=cur_end, channel=0))
                    if i == num_notes - 1:
                        mid.tracks[0].append(mido.Message('note_on', note=cur_mnotes[0], velocity = 0, time = off_dur, channel =0))
                        mid.tracks[0].append(mido.Message('note_off', note=cur_mnotes[0], velocity = 0, time = padding, channel=0))

                mid.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
                um.save_midi(mid,outname)
