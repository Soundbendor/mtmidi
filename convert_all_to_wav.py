import util as um
import os

midis = um.get_saved_midi()
for midi in midis:
   um.write_to_wav(midi)
