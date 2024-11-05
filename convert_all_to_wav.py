import util as um
import os

out_dir = '/media/dxk/TOSHIBA EXT/wav'
midis = um.get_saved_midi()
for im,midi in enumerate(midis):
    um.write_to_wav(midi, save_dir=out_dir)
