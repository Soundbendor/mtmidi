import polars as pl
import util as UM
import os

res_folder = UM.by_projpath(subpath='res_csv', make_dir = False)

df = None
datasets = ['polyrhythms', 'dynamics', 'chords7', 'secondary_dominant', 'modemix_chordprog']

emb_types=['mg_audio','mg_small_h','mg_med_h','mg_large_h','mg_small_at','mg_med_at','mg_large_at','jukebox']

ds_order = {k:i for (i,k) in enumerate(datasets)}
emb_order = {k:i for (i,k) in enumerate(emb_types)}

cols = ['dataset', 'embedding_type',  'toml_file',  'accuracy_score', 'f1_macro', 'f1_micro', 'best_trial_layer_idx', 'eval_valid_score',  'num_trials', 'layer_idx', 'is_classification', 'train_on_middle', 'do_regression_classification', 'to_nep', 'classify_by_subcategory', 'prefix',  'eval', 'debug', 'num_epochs', 'batch_size', 'prune', 'grid_search', 'save_intermediate_model', 'memmap', 'slurm_job', 'thresh', 'model_type', 'model_layer_dim', 'out_dim',  'confmat', 'confmat_path', 'best_trial_obj_value', 'best_trial_dropout',  'best_lr_exp', 'best_weight_decay_exp']
 
for i,_f in enumerate(os.listdir(res_folder)):
    #print(_f)
    cur_dir = os.path.join(res_folder, _f)
    if i == 0:
        df = pl.read_csv(cur_dir)[cols]
    else:
        cur_df = pl.read_csv(cur_dir)
        cur_cols = set(cur_df.columns)
        cur_diff = set(cols).difference(cur_cols)
        if len(cur_diff) > 0:
            for _c in cur_diff:
                cur_df = cur_df.with_columns(**{_c: -1.0})
        #print(cur_df)
        if cur_df['best_trial_layer_idx'].dtype == pl.String:
            cur_df = cur_df.with_columns(best_trial_layer_idx =  -1)
            cur_df = cur_df.cast({'best_trial_layer_idx': pl.Int64})
        df = df.extend(cur_df[cols])

df = df.sort(pl.col('dataset').replace_strict(ds_order), pl.col('embedding_type').replace_strict(emb_order), descending=[False, False])

df.write_csv(os.path.join(res_folder, 'overall.csv'))
