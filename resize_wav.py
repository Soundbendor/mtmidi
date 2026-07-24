import polars as pl
import librosa as lr

def polyrhythms_get_offset(name):
    cur_name = name.split('-')
    offms_substr = cur_name[-2]
    cur_offset = int(offms_substr.split('_')[1])
    return cur_offset

def polyrhythms_get_zero_offset_name(name):
    cur_name = name.split('-')
    cur_name[-2] = 'offms_0'
    ret = '-'.join(cur_name)
    return ret

def dynamics_get_offset(name):
    cur_name = name.split('_')
    offset_substr = cur_name[-1]
    offset = int(offset_substr[3:])
    return offset

def dynamics_get_zero_offset_name(name):
    cur_name = name.split('_')
    cur_name[-1] = 'off0'
    ret = '_'.join(cur_name)
    return ret
