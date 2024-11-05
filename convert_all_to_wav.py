import util as um
import os

out_dir = '/media/dxk/TOSHIBA EXT/wav'
midis = um.get_saved_midi()
for midi in midis:
    um.write_to_wav(midi, save_dir=out_dir)
