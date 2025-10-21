import csv, os, sys, time
import polars as pl

#- 6130 gridsearch 10 boredom folds  root inv one layer
#- 6150 gridsearch 10 boredom folds reduced search root inv one layer
#- 6131 gridsearch 10 boredom folds  1st inv one layer
#- 6151 gridsearch 10 boredom folds reduced search 1st inv one layer
#- 6132 gridsearch 10 boredom folds  2nd inv one layer
#- 6152 gridsearch 10 boredom folds reduced search 2nd inv one layer
#- 6133 gridsearch 10 boredom folds  3rd inv one layer
#- 6153 gridsearch 10 boredom folds reduc
pfix_2l = set([39,33,59,55])
pfix_1l = set([139,133,159,155])
pfix_i0 = set([6150,6130])
pfix_i1 = set([6151,6131])
pfix_i2 = set([6152,6132])
pfix_i3 = set([6153,6133])
pfix_inv = set([6150,6130,6151,6131,6152,6132,6153,6133])
models = ['baseline_concat', 'baseline_chroma', 'baseline_mfcc', 'baseline_mel',  'mg_audio', 'mg_small_h', 'mg_med_h', 'mg_large_h', 'jukebox']
models = models[::-1]
models_li = ['jukebox', 'mg_large_h', 'mg_med_h', 'mg_small_h']
datasets = ['polyrhythms', 'dynamics', 'chords7', 'modemix_chordprog', 'secondary_dominant']
datasets2 = ['chords']
invs = [0,1,2,3]
res = {}
res_f1macro = {}
res_f1micro = {}
res_li = {}
res2 = {}
res2_f1macro = {}
res2_f1micro = {}
res2_li = {}
res_1l = {}
res_1l_f1macro = {}
res_1l_f1micro = {}
res_1l_li = {}
res_inv = {}
res_inv_f1macro = {}
res_inv_f1micro = {}
res_inv_li = {}

res['model'] = [m for m in models]
res_f1macro['model'] = [m for m in models]
res_f1micro['model'] = [m for m in models]
res_li['model'] = [m for m in models_li]
res2['model'] = [m for m in models]
res2_f1macro['model'] = [m for m in models]
res2_f1micro['model'] = [m for m in models]
res2_li['model'] = [m for m in models_li]
res_1l['model'] = [m for m in models]
res_1l_f1macro['model'] = [m for m in models]
res_1l_f1micro['model'] = [m for m in models]
res_1l_li['model'] = [m for m in models_li]
res_inv['model'] = [m for m in models]
res_inv_f1macro['model'] = [m for m in models]
res_inv_f1micro['model'] = [m for m in models]
res_inv_li['model'] = [m for m in models_li]


for d in datasets:
    res[d] = [-1.0 for _ in range(len(models))]
    res_f1macro[d] = [-1.0 for _ in range(len(models))]
    res_f1micro[d] = [-1 for _ in range(len(models))]
    res_li[d] = [-1.0 for _ in range(len(models_li))]
    res_1l[d] = [-1.0 for _ in range(len(models))]
    res_1l_f1macro[d] = [-1.0 for _ in range(len(models))]
    res_1l_f1micro[d] = [-1.0 for _ in range(len(models))]
    res_1l_li[d] = [-1 for _ in range(len(models_li))]

for d in datasets2:
    res2[d] = [-1.0 for _ in range(len(models))]
    res2_f1macro[d] = [-1.0 for _ in range(len(models))]
    res2_f1micro[d] = [-1.0 for _ in range(len(models))]
    res2_li[d] = [-1 for _ in range(len(models_li))]

for inv in invs:
    res_inv[f'inv_{inv}'] = [-1.0 for _ in range(len(models))]
    res_inv_f1macro[f'inv_{inv}'] = [-1.0 for _ in range(len(models))]
    res_inv_f1micro[f'inv_{inv}'] = [-1.0 for _ in range(len(models))]
    res_inv_li[f'inv_{inv}'] = [-1 for _ in range(len(models_li))]

m_idx = {m:i for (i,m) in enumerate(models)}
mli_idx = {m:i for (i,m) in enumerate(models_li)}
cur_schema = [("model", pl.String)] + [(d, pl.Float64) for d in datasets]
cur_schema2 = [("model", pl.String)] + [(d, pl.Float64) for d in datasets2]
cur_schema_inv = [('model', pl.String)] + [(f'inv_{inv}', pl.Float64) for inv in invs]
cur_time = int(time.time() * 1000)
outfname = f'combined_result.csv'
outfname_f1macro = f'combined_result_f1macro.csv'
outfname_f1micro = f'combined_result_f1micro.csv'
outfname_li = f'combined_result_li.csv'
outfname2 = f'combined_result2.csv'
outfname2_f1macro = f'combined_result2_f1macro.csv'
outfname2_f1micro = f'combined_result2_f1micro.csv'
outfname2_li = f'combined_result2_li.csv'
outfname_1l = f'combined_result_1l.csv'
outfname_1l_f1macro = f'combined_result_1l_f1macro.csv'
outfname_1l_f1micro = f'combined_result_1l_f1micro.csv'
outfname_1l_li = f'combined_result_1l_li.csv'
outfname_inv = f'combined_result_inv.csv'
outfname_inv_f1macro = f'combined_result_inv_f1macro.csv'
outfname_inv_f1micro = f'combined_result_inv_f1micro.csv'
outfname_inv_li = f'combined_result_inv_li.csv'


out_file = os.path.join('res_csv',outfname)
out_file_f1macro = os.path.join('res_csv',outfname_f1macro)
out_file_f1micro = os.path.join('res_csv',outfname_f1micro)
out_file_li = os.path.join('res_csv',outfname_li)
out_file2 = os.path.join('res_csv',outfname2)
out_file2_f1macro = os.path.join('res_csv',outfname2_f1macro)
out_file2_f1micro = os.path.join('res_csv',outfname2_f1micro)
out_file2_li = os.path.join('res_csv',outfname2_li)
out_file_1l = os.path.join('res_csv',outfname_1l)
out_file_1l_f1macro = os.path.join('res_csv',outfname_1l_f1macro)
out_file_1l_f1micro = os.path.join('res_csv',outfname_1l_f1micro)
out_file_1l_li = os.path.join('res_csv',outfname_1l_li)
out_file_inv = os.path.join('res_csv',outfname_inv)
out_file_inv_f1macro = os.path.join('res_csv',outfname_inv_f1macro)
out_file_inv_f1micro = os.path.join('res_csv',outfname_inv_f1micro)
out_file_inv_li = os.path.join('res_csv',outfname_inv_li)
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
            if cur_prefix in pfix_2l:
                if res[cur_ds][cur_idx] < 0.0:
                    res[cur_ds][cur_idx] = float(row['accuracy_score'])
                    res_f1macro[cur_ds][cur_idx] = float(row['f1_macro'])
                    res_f1micro[cur_ds][cur_idx] = float(row['f1_micro'])
                if cur_emb in models_li:
                    cur_idx = mli_idx[cur_emb]
                    if res_li[cur_ds][cur_idx] < 0:
                        res_li[cur_ds][cur_idx] = int(row['best_trial_layer_idx'])
            elif cur_prefix in pfix_1l:
                if res_1l[cur_ds][cur_idx] < 0.0:
                    res_1l[cur_ds][cur_idx] = float(row['accuracy_score'])
                    res_1l_f1macro[cur_ds][cur_idx] = float(row['f1_macro'])
                    res_1l_f1micro[cur_ds][cur_idx] = float(row['f1_micro'])

                if cur_emb in models_li:
                    cur_idx = mli_idx[cur_emb]
                    if res_1l_li[cur_ds][cur_idx] < 0:
                        res_1l_li[cur_ds][cur_idx] = int(row['best_trial_layer_idx'])
            elif cur_prefix in pfix_inv:
                cur_inv = cur_prefix % 10
                cur_inv_str = f'inv_{cur_inv}'
                if res_inv[cur_inv_str][cur_idx] < 0.0:
                    res_inv[cur_inv_str][cur_idx] = float(row['accuracy_score'])
                    res_inv_f1macro[cur_inv_str][cur_idx] = float(row['f1_macro'])
                    res_inv_f1micro[cur_inv_str][cur_idx] = float(row['f1_micro'])

                if cur_emb in models_li:
                    cur_idx = mli_idx[cur_emb]
                    if res_inv_li[cur_inv_str][cur_idx] < 0:
                        res_inv_li[cur_inv_str][cur_idx] = int(row['best_trial_layer_idx'])
                    
        else:
            if res2[cur_ds][cur_idx] < 0.0:
                res2[cur_ds][cur_idx] = float(row['accuracy_score'])
                res2_f1macro[cur_ds][cur_idx] = float(row['f1_macro'])
                res2_f1micro[cur_ds][cur_idx] = float(row['f1_micro'])

            if cur_emb in models_li:
                cur_idx = mli_idx[cur_emb]
                if res2_li[cur_ds][cur_idx] < 0:
                    res2_li[cur_ds][cur_idx] = int(row['best_trial_layer_idx'])
    iptf.close()

print(res)
print(res2)
print(res_inv)
df = pl.DataFrame(res, schema=cur_schema)
df_f1macro = pl.DataFrame(res_f1macro, schema=cur_schema)
df_f1micro = pl.DataFrame(res_f1micro, schema=cur_schema)
df_li = pl.DataFrame(res_li, schema=cur_schema)
df2 = pl.DataFrame(res2, schema=cur_schema2)
df2_f1macro = pl.DataFrame(res2_f1macro, schema=cur_schema2)
df2_f1micro = pl.DataFrame(res2_f1micro, schema=cur_schema2)
df2_li = pl.DataFrame(res2_li, schema=cur_schema2)
df_1l = pl.DataFrame(res_1l, schema=cur_schema)
df_1l_f1macro = pl.DataFrame(res_1l_f1macro, schema=cur_schema)
df_1l_f1micro = pl.DataFrame(res_1l_f1micro, schema=cur_schema)
df_1l_li = pl.DataFrame(res_1l_li, schema=cur_schema)

df_inv = pl.DataFrame(res_inv, schema=cur_schema_inv)
df_inv_f1macro = pl.DataFrame(res_inv_f1macro, schema=cur_schema_inv)
df_inv_f1micro = pl.DataFrame(res_inv_f1micro, schema=cur_schema_inv)
df_inv_li = pl.DataFrame(res_inv_li, schema=cur_schema_inv)

df.write_csv(out_file, separator=",")
df_f1macro.write_csv(out_file_f1macro, separator=",")
df_f1micro.write_csv(out_file_f1micro, separator=",")
df_li.write_csv(out_file_li, separator=",")
df2.write_csv(out_file2, separator=",")
df2_f1macro.write_csv(out_file2_f1macro, separator=",")
df2_f1micro.write_csv(out_file2_f1micro, separator=",")
df2_li.write_csv(out_file2_li, separator=",")
df_1l.write_csv(out_file_1l, separator=",")
df_1l_f1macro.write_csv(out_file_1l_f1macro, separator=",")
df_1l_f1micro.write_csv(out_file_1l_f1micro, separator=",")
df_1l_li.write_csv(out_file_1l_li, separator=",")
df_inv.write_csv(out_file_inv, separator=",")
df_inv_f1macro.write_csv(out_file_inv_f1macro, separator=",")
df_inv_f1micro.write_csv(out_file_inv_f1micro, separator=",")
df_inv_li.write_csv(out_file_inv_li, separator=",")
