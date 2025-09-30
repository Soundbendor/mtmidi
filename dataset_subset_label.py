import util_data as UD
import util as UM
import os,sys,time,argparse
import tomllib
from distutils.util import strtobool
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold 

seed = 5

csv_folder = UM.by_projpath(subpath='csv', make_dir = False)
hf_csv_folder = UM.by_projpath(subpath='hf_csv', make_dir = False)
def check_overlap(idx_arrs):
    num_folds = len(idx_arrs)
    overlap = False
    for i in range(num_folds):
        for j in range(i+1,num_folds):
            fold1 = set(idx_arrs[i])
            fold2 = set(idx_arrs[j])
            if len(fold1.intersection(fold2)) > 0:
                overlap = True
                break
    return overlap

def check_cover(idx_arrs, num_ex):
    x = set([])
    num_folds = len(idx_arrs)
    for i in range(num_folds):
        x = x.union(idx_arrs[i])
    idx_set = set(range(num_ex))
    return len(idx_set.difference(x)) == 0

def check_counts(df, fold_col):
    fold_counts = df[fold_col].value_counts().sort(fold_col).to_dicts()
    col_dict = {}
    for fold in fold_counts:
        fold_name = fold[fold_col]
        fold_count = fold['count']
        print(f'==== fold {fold_name} : {fold_count}====')
        cur_f = df.filter(pl.col(fold_col) == fold_name)
        for ic, col in enumerate(df.columns):
            if col not in ['fold', 'set_type']:
                cur_vc = cur_f[col].value_counts().sort(col)
                num_vc = len(cur_vc)
                cur_vc = cur_vc.with_columns(proportion = pl.col('count') / fold_count)
                cur_vc = cur_vc.insert_column(0, pl.Series(f'{fold_col}_count', [fold_count] * num_vc))
                cur_vc = cur_vc.insert_column(0, pl.Series(fold_col, [fold_name] * num_vc))
                if col not in col_dict.keys():
                    col_dict[col] = cur_vc
                else:
                    col_dict[col] = col_dict[col].extend(cur_vc)
    return col_dict

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-fd", "--folds", type=strtobool, default=True, help="folds (True) or train_test_split")

    args = parser.parse_args()
    arg_dict = vars(args)
    cur_dsname = arg_dict['dataset'] 
    want_folds = arg_dict['folds']
    out_csvf = f'{cur_dsname}-folds.csv'
    cur_dd = UD.load_data_dict(cur_dsname, classify_by_subcategory = True, tomlfile_str = '', use_folds = False)
    df = cur_dd['df']
    num_ex = len(df)
    #print(num_ex)
    ex_idxs = np.arange(num_ex)
    label_col = cur_dd['label_col']
    #labels = df[label_col]
    label_arr = cur_dd['label_arr']
    classdict = cur_dd['pl_classdict']
    #print(num_ex, len(label_idxs))

    if want_folds == True:
        out_folder = None
        folds = None
        if cur_dd['is_hf'] == False:
            out_folder = UM.by_projpath(subpath='fold_data', make_dir = True) 
        else:
            out_folder = UM.by_projpath(subpath='hf_fold_data', make_dir = True) 
        if cur_dsname != 'tempos':
            label_idxs = np.array([classdict[x] for x in label_arr])
            skf = StratifiedKFold(n_splits = 20, random_state = seed, shuffle = True)
            #print(skf.get_n_splits(num_ex, label_idxs))
            folds = [y for (x,y) in skf.split(ex_idxs, label_idxs)]
        else:
            # folds 1-14 are middle bpms, do stratified split (14)
            # folds 15-20 are from top and bottom bpms, do stratified split (6)
            num_samples = cur_dd['df'].shape[0]
            sort_df = cur_dd['df'].sort('bpm')
            props = [0., 0.15, 0.85, 1.]
            cutoffs = [int(num_samples * c) for c in props]
            idxs = [np.arange(cutoffs[c], cutoffs[c+1]) for c in range(len(props)-1)]
            train_idxs = idxs[1]
            train_labels = sort_df[train_idxs]['bpm'].to_numpy()
            validtest_idxs = np.hstack((idxs[0], idxs[2]))
            validtest_labels = sort_df[validtest_idxs]['bpm'].to_numpy()
            
            # hard to do stratified split, just do kfold split
            skf = StratifiedKFold(n_splits = 14, random_state = seed, shuffle = True)
            skf2 = StratifiedKFold(n_splits = 6, random_state = seed, shuffle = True)
            train_folds = [train_idxs[y] for (x,y) in skf.split(train_idxs, train_labels)]
            validtest_folds = [validtest_idxs[y] for (x,y) in skf2.split(validtest_idxs, validtest_labels)]
            folds = train_folds + validtest_folds
 
        fold_col = np.zeros(num_ex, dtype=int)
        set_type_col = np.repeat('train', num_ex).astype(str)
        for i,f in enumerate(folds):
            real_idx = i+1
            fold_col[f] = real_idx
            if real_idx >= 1 and real_idx <= 14:
                set_type_col[f] = 'train'
            elif real_idx >= 15 and real_idx <= 17:
                set_type_col[f] = 'valid'
            else:
                set_type_col[f] = 'test'

        has_overlap = check_overlap(folds)
        covers = check_cover(folds, num_ex)
        print(f'overlaps: {has_overlap}, covers: {covers}')
        df = df.with_columns(**{'fold': fold_col, 'set_type': set_type_col})
        #print(df['fold'].unique_values)
        col_dict = check_counts(df, 'fold')
        for col_name, col_df in col_dict.items():
            cur_fname = f'{cur_dsname}-{col_name}-byfold.csv'
            cur_fpath = os.path.join(out_folder, cur_fname)
            col_df.write_csv(cur_fpath)
        
        col_dict2 = check_counts(df, 'set_type')
        for col_name, col_df in col_dict2.items():
            cur_fname = f'{cur_dsname}-{col_name}-bysettype.csv'
            cur_fpath = os.path.join(out_folder, cur_fname)
            col_df.write_csv(cur_fpath)

        if cur_dd['is_hf'] == False:
            df.write_csv(os.path.join(csv_folder, out_csvf))
        else:
            df.write_csv(os.path.join(hf_csv_folder, out_csvf))
        #print(np.alltrue(fold_col > 0))
        #for i, (tr_idx, te_idx) in enumerate(skf.split(ex_idxs, label_idxs)):
        #print(i, len(tr_idx), len(te_idx))

    #print(label_idxs)


    #print(df)
