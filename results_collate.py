import csv, os, sys, time
import polars as pl



models = ['baseline_mel', 'baseline_mfcc', 'baseline_chroma', 'baseline_concat', 'mg_audio', 'mg_small_h', 'mg_med_h', 'mg_large_h', 'jukebox']
datasets = ['polyrhythms', 'dynamics', 'chords7', 'modemix_chordprog', 'secondary_dominant']
datasets2 = ['chords']
res = {}
res2 = {}
res['dataset'] = [d for d in datasets]
res2['dataset'] = [d for d in datasets2]
for m in models:
    res[m] = [-1.0 for _ in range(len(datasets))]
    res2[m] = [-1.0 for _ in range(len(datasets2))]
d_idx = {d:i for (i,d) in enumerate(datasets)}
d2_idx = {d:i for (i,d) in enumerate(datasets2)}
cur_schema = [("dataset", pl.String)] + [(m, pl.Float64) for m in models]
cur_time = int(time.time() * 1000)
outfname = f'combined_result.csv'
outfname2 = f'combined_result2.csv'
out_file = os.path.join('res_csv',outfname)
out_file2 = os.path.join('res_csv',outfname2)
for fi,f in enumerate(os.listdir('res_csv')):
    if 'combined' in f:
        continue
    iptf = open(os.path.join('res_csv', f), 'r')
    cur_split = f.split("-")
    cur_prefix = int(cur_split[0])
    cur_ds = cur_split[1]
    cur_emb = cur_split[2].split(".")[0]
    cur_idx = -1
    new_dataset = cur_ds in datasets
    if new_dataset == True:
        cur_idx = d_idx[cur_ds]
    else:
        cur_idx = d2_idx[cur_ds]
    csvr = csv.DictReader(iptf)
    for row in csvr:
        if new_dataset == True:
            if res[cur_emb][cur_idx] < 0.0:
                res[cur_emb][cur_idx] = float(row['accuracy_score'])
        else:
            if res2[cur_emb][cur_idx] < 0.0:
                res2[cur_emb][cur_idx] = float(row['accuracy_score'])
    iptf.close()

print(res)
print(res2)
df = pl.DataFrame(res, schema=cur_schema)
df2 = pl.DataFrame(res2, schema=cur_schema)
df.write_csv(out_file, separator=",")
df2.write_csv(out_file2, separator=",")
