import util as UM
import mido
import musicnoteconv as mnc
from itertools import combinations
import dynamics as DYN
import os,csv
#instruments = ['Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum']
midinote = 60 # midi note to use
ticks_per_beat = 1000
tempo_microsec = mido.bpm2tempo(DYN.bpm)
sustain = 1.0
beg_padding = 0
end_padding = 10
bpm = 60

# copied from midi_make_polyrhythms
tick_offsets = {x:(offset_lvl, int(UM.ms_to_ticks(x,ticks_per_beat = ticks_per_beat, bpm = bpm))) for offset_lvl, x in enumerate(DYN.offset_ms_arr)}

csvpath = os.path.join(UM.by_projpath('csv'), 'dynamics.csv')
outf = open(csvpath, 'w')
csvw = csv.DictWriter(outf,fieldnames=DYN.fieldnames)
csvw.writeheader()


# always goes from soft to loud
#dyn_to_use = dyn_ltr
for _bsd in DYN.beat_subdiv_arr:
    beat = _bsd[0]
    subdiv = _bsd[1]
    notes_per_bar = beat * subdiv
    total_notes = notes_per_bar * DYN.num_bars
    for offset_ms, (offset_lvl, offset_ticks) in tick_offsets.items():
        # don't do offsets bigger than 0 for now
        if offset_ms > 0:
            break
        for _dt in DYN.dyn_types:
            # since calculating velocites from first dynamic
            # flat needs to calculate using dyn1 as first dynamic and dyn2 as first dynamic
            # so repeat the whole loop flipping the dynamics
            num_times = 1
            if _dt == 'flat':
                num_times = 2
            inflection_pt = 0
            dyn_category = _dt # this is the big category (different for dynamics with inflection point)
            dyn_subcategory = _dt # make this the same as dyn_category for non-inflection point dynamics
            has_inflection_pt = DYN.has_inflection_pt(_dt)
            # if has inflection point, get it
            if has_inflection_pt == True:
                dyn_category, inflection_pt = DYN.get_dyn_category_and_inflection_pt(_dt)
            for rvb_lvl, rvb_val in UM.reverb_lvl.items():
                for cur_time in range(num_times):
                    for _dp in DYN.dyn_pairs:
                        dyn1 = _dp[0]
                        dyn2 = _dp[1]
                        # first time around for 'flat', make both dynamics same
                        if _dt == 'flat':
                            dyn2 = _dp[0]
                        # second time around for 'flat'
                        # go ahead and make both dyn1 and dyn2 the same since ignoring second
                        if cur_time == 1:
                            dyn1 = _dp[1]
                            dyn2 = _dp[1]
                        _v = dyn.get_velocities(dyn1, dyn2, beat_subdiv = _bsd, start_beat_dyn2 = inflection_pt, dyn_category=dyn_category)
                        for _inst in dyn.instruments:
                            pg_num = UM.get_inst_program_number(_inst)
                            midi_num = UM.get_inst_midinote(_inst, default=midinote)
                            is_drum = UM.is_inst_drum(_inst)
                            short_name = ''.join(_inst.split(' '))
                            chnum = 0 if is_drum == false else UM.drum_chnum
                            mid = none
                            mid = mido.midifile(type=1, ticks_per_beat=ticks_per_beat)
                            mid.tracks.append(mido.miditrack())
                            mid.tracks[0].append(mido.metamessage('set_tempo', tempo = tempo_microsec, time = 0))

                            mid.tracks[0].append(mido.message('control_change', control=91, value=rvb_val, time =0, channel=chnum))
                            mid.tracks[0].append(mido.message('program_change', program=_pgnum, time =0, channel=chnum))
                            mid.tracks[0].name = short_name
                            d_on, d_off = UM.notedur_to_ticks(dur, subdiv = subdiv, ticks_per_beat = ticks_per_beat, sustain = sustain)
                            dynstr = f"{_dt}-{dyn1}_{dyn2}"
                            durstr = f"{dur}_{subdiv}"
                            outname = f"dyn-{short_name}-{dynstr}-{durstr}.mid"
                            outname = DYN.get_outname(dyn_subcategory, dyn1, dyn2, short_name, beat, subdiv, rvb_lvl, offset_ms, ext="mid")
                            cur_row = {'dyn1': dyn1, 'dyn2': dyn2,  'inst': short_name, 'inflection_point': inflection_point,
                                       'dyn_category': dyn_category, 'dyn_subcategory': dyn_subcategory,
                                       'rvb_lvl': rvb_lvl, 'rvb_val':rvb_val,
                                       'offset_lvl':offset_lvl, 'offset_ms': offset_ms, 'offset_ticks': offset_ticks,
                                       'beats_per_bar':beat, 'num_bars':DYN.num_bars,
                                       'beat_subdiv': subdiv, 'bpm': bpm, 'num_beats': beat * DYN.num_bars}
                            
                            csvw.writerow(cur_row)
                            #print(outname)
                            # keep track of ticks left from last bar
                            last_ticks_left = 0
                            for curbar in range(DYN.num_bars):
                                # keep track of ticks left over
                                ticks_left = ticks_per_beat * 4 # 4/4 bar
                                for notenum in range(notes_per_bar):
                                    first_beat = (curbar == 0) and notenum == 0
                                    start_time = d_off
                                    realnum = notenum + (curbar * notes_per_bar)
                                    curvel = _v[realnum]
                                    if notenum == 0:
                                        start_time = 0
                                    
                                    # not sure of cur padding/offset_scenario since copied from midi_make_polyrhythms
                                    # might be hacky
                                    cur_padding = (beg_padding + offset_ticks) if first_beat == True else 0
                                    mid.tracks[0].append(mido.message('note_on', note=midinote, velocity=curvel, time=start_time + last_ticks_left + cur_padding, channel=chnum))
                                    ticks_left -= start_time
                                    last_ticks_left = 0
                                    end_time = d_on
                                    mid.tracks[0].append(mido.message('note_off', note=midinote, velocity=curvel, time=end_time, channel=chnum))
                                    ticks_left -= end_time
                                    if notenum == (notes_per_bar) - 1:
                                        last_ticks_left = max(0,ticks_left)
                            if end_padding > 0 or last_ticks_left > 0:
                                if end_padding > 0:
                                    # only delay note_on by last ticks left, delay note_off by end_padding
                                    # so total is last ticks left + end_padding
                                    mid.tracks[0].append(mido.message('note_on', note=midinote, velocity=0, time=last_ticks_left, channel=chnum))
                                    mid.tracks[0].append(mido.message('note_off', note=midinote, velocity=0, time=end_padding, channel=chnum))
                                else:
                                    # don't delay note on, delay note off by last ticks left
                                    mid.tracks[0].append(mido.message('note_on', note=midinote, velocity=0, time=0,channel=chnum))
                                    mid.tracks[0].append(mido.message('note_off', note=midinote, velocity=0, time=end_padding, channel=chnum))
                            mid.tracks[0].append(mido.metamessage('end_of_track', time=0))
                            UM.save_midi(mid, outname, dataset="dynamics")

outf.close()


                        
