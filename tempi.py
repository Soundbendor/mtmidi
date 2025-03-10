import numpy as np

global classdict
global rev_classdict
global classlist_sorted
global classset_aug
global num_classes
global bpm_class_mapper


minbpm = 50
maxbpm = 210
bpmrange = maxbpm - minbpm
default_cls = -1

def get_class_medians(class_binsize):
    _bpm_class_mapper = lambda x: int((x - minbpm)/class_binsize)
    half_binsize = class_binsize//2 
    # medians of bpm bin to index
    d = { (i + half_binsize): _bpm_class_mapper(i) for i in range(minbpm, maxbpm + class_binsize, class_binsize)}
    # medians
    dlist = [(i + half_binsize) for i in range(minbpm, maxbpm + class_binsize, class_binsize)]

    cur_num_classes = len(dlist)
    # add nullclass (don't want in list of classes)
    d[default_cls] = cur_num_classes # maps to the very last clsas
    # index to bpm bin
    revd = {x:i for (i,x) in d.items()}

    dset_aug = set(dlist + [default_cls])
    return d, revd, dlist, dset_aug, cur_num_classes, _bpm_class_mapper


def get_nearest_bpmclass(normed_pred, sorted_bpm_list, thresh=3):
    pred = (normed_pred * bpmrange) + minbpm
    _start_idx = 0
    _end_idx = len(sorted_bpm_list)
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

def init(class_binsize=3):

    global classdict
    global rev_classdict
    global classlist_sorted
    global classset_aug
    global num_classes
    global bpm_class_mapper
    # classdict maps middles of bpm bins to indices (+ a default class for non matches)
    # rev_classdict maps indices to middles of bpm bins
    # classlist_sorted lists all classes (without default class)
    # classset_aug is a set combining classlist_sorted and default_class

    classdict, rev_classdict, classlist_sorted, classset_aug, num_classes, bpm_class_mapper = get_class_medians(class_binsize)
