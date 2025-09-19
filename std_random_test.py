# checking coverage of random sample
# using the method of https://github.com/brown-palm/syntheory/blob/4f222359e750ec55425c12809c1a0358b74fce49/probe/probes.py#L146

import numpy as np
import random

random.seed(5)

bs = 256

num_epochs = 100

ds_size = 45000
seen_idxs = set([])
dist = np.zeros(ds_size).astype(np.int64)
step = 0
while True:
    epoch = ( step * bs)/float(ds_size)
    idxs = random.sample(range(ds_size), min(bs, ds_size))
    idx_set = set(idxs)
    cur_intersect = seen_idxs.intersection(idx_set)
    
    for idx in idxs:
        dist[idx] += 1
    """
    if len(cur_intersect) > 0:
        print(cur_intersect)
        print(len(cur_intersect), epoch)
    """
    seen_idxs = seen_idxs.union(idxs)
    step += 1
    if epoch >= num_epochs:
        break

min_idx = np.argmin(dist)
max_idx = np.argmax(dist)

minmax = f'min: ({min_idx} {dist[min_idx]}), max: ({max_idx}, {dist[max_idx]})'

print(minmax)
with open(f'sample_counts-{bs}-{num_epochs}.txt', 'w') as f:
    for i in range(ds_size):
        f.write(f'{i},{dist[i]}\n')

with open(f'sample_minmax-{bs}-{num_epochs}.txt', 'w') as f:
    f.write(minmax)
