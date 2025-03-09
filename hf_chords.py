quality_to_idx = {'major': 0, 'minor': 1, 'aug': 2, 'dim': 3}
idx_to_quality = {y:x for (x,y) in quality_to_idx.items()}
num_chords = len(quality_to_idx.keys())
