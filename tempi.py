import numpy as np


minbpm = 50
maxbpm = 210
bpmrange = maxbpm - minbpm
def get_class_medians(class_binsize):
    bpmclass_mapper = lambda x: int((x - minbpm)/class_binsize)
    half_binsize = class_binsize//2 
    # medians of bpm bin to index
    d = { (i + half_binsize): bpmclass_mapper(i) for i in range(minbpm, maxbpm + class_binsize, class_binsize)}
        # medians
    dlist = [(i + half_binsize) for i in range(minbpm, maxbpm + class_binsize, class_binsize)]

    num_classes = len(dlist)
    # add nullclass (don't want in list of classes)
    d[-1] = num_classes # maps to the very last clsas
    # index to bpm bin
    revd = {x:i for (i,x) in d.items()}

     
    return d, revd, dlist


def get_nearest_bpmclass(normed_pred, sorted_bpm_list, thresh=3):
    pred = (normed_pred * bpmrange) + minbpm
    _start_idx = 0
    _end_idx = len(poly_tups_norm)
    def _get_nearest(start_idx, end_idx, ipt):
        mid_idx = (end_idx+start_idx)//2
        mid_val = sorted_bpm_list[mid_idx]
        match = np.isclose(ipt, mid_val, atol=thresh)
        #print(f"idx0: {start_idx}, idx1:{end_idx}, ipt:{ipt}, mididx:{mid_idx}, midval:{mid_val}, match:{match}")
        _ret = -1
        if match == True:
            _ret = sorted_bpm_list[mid_idx]
        else:
            if ipt < mid_val:
                start = start_idx
                end = mid_idx
                if end >start:
                    #print(f'recurse left, start:{start}, end:{end}')
                    _ret = _get_nearest(start,end, ipt)
            else:
                start = mid_idx+1
                end = end_idx
                if end >start:
                    #print(f'recurse right, start:{start}, end:{end}')
                    _ret = _get_nearest(start,end, ipt)
        return _ret
    ret = _get_nearest(_start_idx, _end_idx, normed_pred)
    return ret


