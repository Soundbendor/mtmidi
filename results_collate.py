import csv, os, sys, time
import polars as pl



models = ['baseline_mel', 'baseline_mfcc', 'baseline_chroma', 'baseline_concat', 'mg_audio', 'mg_small_h', 'mg_med_h', 'mg_large_h', 'jukebox']
datasets = ['polyrhythms', 'dynamics', 'chords7', 'modemix_chordprog', 'secondary_dominant']
res = {}
res['dataset'] = [d for d in datasets]
for m in models:
    res[m] = [-1.0 for _ in range(len(datasets))]
d_idx = {d:i for (i,d) in enumerate(datasets)}
cur_schema = [("dataset", pl.String)] + [(m, pl.Float64) for m in models]
cur_time = int(time.time() * 1000)
outfname = f'combined_result.csv'
out_file = os.path.join('res_csv',outfname)
for fi,f in enumerate(os.listdir('res_csv')):
    if 'combined' in f:
        continue
    iptf = open(os.path.join('res_csv', f), 'r')
    cur_split = f.split("-")
    cur_prefix = int(cur_split[0])
    cur_ds = cur_split[1]
    cur_emb = cur_split[2].split(".")[0]
    cur_idx = d_idx[cur_ds]
    csvr = csv.DictReader(iptf)
    for row in csvr:
        if res[cur_emb][cur_idx] < 0.0:
            res[cur_emb][cur_idx] = float(row['accuracy_score'])
    iptf.close()

print(res)
df = pl.DataFrame(res, schema=cur_schema)
df.write_csv(out_file, separator=",")
