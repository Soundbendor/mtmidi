from datasets import load_dataset


# train is a key of tempi with value array
# each element of tempi['train'] has keys (['audio', 'bpm', 'click_config_name', 'midi_program_num', 'offset_time'])
# audio is keyed (['path', 'array', 'sampling_rate'])
# audio.path is filename
#audio.array is array of audio samples?
# audio.sampling_rate is sampling rate
# bpm is int
# click_config_name is name of instrument as str
# midi_program_num is int
# offset_time is float in seconds?

tempi = load_dataset("meganwei/syntheory", "tempos")
"""
for x in tempi['train']:
    print(x)
#print(len(tempi['train']))
for i,x in enumerate(tempi['train']):
    cur = x['audio']
    print(cur['path'], cur['sampling_rate'])
    print(i,cur['array'].shape)
"""
#print(tempi['train'][30])
