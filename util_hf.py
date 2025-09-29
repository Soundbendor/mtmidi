# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
import datasets as HFDS
import numpy as np

# ---- TEMPI (tempos)--- 
# train is a key of tempi with value array
# each element of tempi['train'] has keys (['audio', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time'])
# audio is keyed (['path', 'array', 'sampling_rate'])
# audio.path is filename
#audio.array is array of audio samples?
# audio.sampling_rate is sampling rate given in Hz
# bpm is int
# click_config_name is name of instrument as str
# midi_program_num is int
# offset_time is float in seconds?


def load_syntheory_train_dataset(ds_name, streaming = True):
    cur_ds =  HFDS.load_dataset("meganwei/syntheory", ds_name, split = 'train', streaming = streaming)
    return cur_ds

def get_from_entry_syntheory_audio(cur_entry, mono=True, normalize =True, dur = 4.0):
    #cur_aud = train_ds[idx]['audio']
    cur_aud = cur_entry['audio']
    cur_sr = cur_aud['sampling_rate']
    cur_arr = None
    want_samp = int(np.round(cur_sr * dur))
    if cur_aud['array'].shape[0] > 1:
        cur_arr = np.mean(cur_aud['array'], axis=0)
    else:
        cur_arr = cur_aud['array'].flatten()
    if normalize == True:
        cur_arr = cur_arr/np.max(np.abs(cur_arr))
    return cur_arr[:want_samp], cur_sr


