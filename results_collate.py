import csv, os, sys, time
import polars as pl



models = ['baseline_mel', 'baseline_mfcc', 'baseline_chroma', 'baseline_concat', 'mg_audio', 'mg_small_h', 'mg_med_h', 'mg_large_h', 'jukebox']
models = models[::-1]
datasets = ['polyrhythms', 'dynamics', 'chords7', 'modemix_chordprog', 'secondary_dominant']
datasets2 = ['chords']
res = {}
res2 = {}

res['model'] = [m for m in models]
res2['model'] = [m for m in models]
for d in datasets:
    res[d] = [-1.0 for _ in range(len(models))]

for d in datasets2:
    res2[d] = [-1.0 for _ in range(len(models))]
m_idx = {m:i for (i,m) in enumerate(models)}
cur_schema = [("model", pl.String)] + [(d, pl.Float64) for d in datasets]
cur_schema2 = [("model", pl.String)] + [(d, pl.Float64) for d in datasets2]
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
    cur_idx = m_idx[cur_emb]
    new_dataset = cur_ds in datasets
    csvr = csv.DictReader(iptf)
    for row in csvr:
        if new_dataset == True:
            if res[cur_ds][cur_idx] < 0.0:
                res[cur_ds][cur_idx] = float(row['accuracy_score'])
        else:
            if res2[cur_ds][cur_idx] < 0.0:
                res2[cur_ds][cur_idx] = float(row['accuracy_score'])
    iptf.close()

print(res)
print(res2)
df = pl.DataFrame(res, schema=cur_schema)
df2 = pl.DataFrame(res2, schema=cur_schema2)
df.write_csv(out_file, separator=",")
df2.write_csv(out_file2, separator=",")
