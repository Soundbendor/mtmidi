# Castellon, R., Donahue, C., & Liang, P. (2021). Codified audio language modeling learns useful representations for music information retrieval. arXiv preprint arXiv:2107.05677.

import librosa
import numpy as np
import os
import soundfile as sf
# calm:
# resample to 44.1
# normalize
# codify using jukebox jq-vae
# audio into languagemodel

wav_dir = '/media/dxk/TOSHIBA EXT/wav'
#wav_dir = "wav"
wav2_dir = '/media/dxk/TOSHIBA EXT/wav_trim2'
#wav2_dir = '/media/dxk/TOSHIBA EXT/wav_trim'
#wav2_dir = "wav_trim"

sr = 32000
want_sec = 4
want_samp = sr * want_sec
print(want_samp)

if os.path.exists(wav2_dir) == False:
    os.makedirs(wav2_dir)

for iw,x in enumerate(os.listdir(wav_dir)):
    #if iw > 0: break
    wpath = os.path.join(wav_dir, x)
    snd, load_sr = librosa.load(wpath, mono = True, sr=sr)
    snd_trim = snd[:want_samp]
    wsplit = x.split('.')
    wjoin = '.'.join(wsplit[:-1]) + '-trim_en.wav'
    opath = os.path.join(wav2_dir, wjoin)
    sf.write(opath, snd_trim, sr)
    
