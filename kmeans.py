import numpy as np
from distutils.util import strtobool
import util_data as UD
import polars as pl
import argparse
import sklearn.cluster as SKC


seed = 5

def get_cluster_composition(cur_ds,cur_df,cur_labelcol, num_clusters,cur_clustering):
    unique_labels = cur_df[cur_labelcol].unique().to_numpy().flatten()
    default_props = {k: 0.0 for k in unique_labels}
    cluster_series = pl.Series('cluster', cur_clustering)
    cur_df.insert_column(-1, cluster_series)
    res = [None for x in range(num_clusters)]
    for c_idx in range(num_clusters):
        cur_props = cur_df.filter(pl.col('cluster') == 0)[cur_labelcol].value_counts(normalize=True,sort=True)
        cur_max = cur_props[0]
        max_label = cur_max[cur_labelcol]
        max_prop = cur_max['proportion']
        cur_res = default_props | cur_props.rows_by_key(key=[cur_labelcol],unique=True)
        res[c_idx] = cur_res | {'max_label': max_label, 'max_proportion': max_prop}
    return res

    

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-li", "--layer_idx", type=int, default=0, help="specifies layer_idx 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=False, help="train on middle")
    parser.add_argument("-rc", "--do_regression_classification", type=strtobool, default=False, help="do regression classification")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    
    args = parser.parse_args()
    arg_dict = vars(args)

    cur_embtype = arg_dict['embedding_type']
    layer_idx = arg_dict['layer_idx']
    cur_dsname = arg_dict['dataset']
    tomlfile_str = arg_dict['toml_file']
    cls_subcat = arg_dict['classify_by_subcategory']
    cur_tom = arg_dict['train_on_middle']
    cur_mm = arg_dict['memmap']
    save_ext = 'npy'
    if cur_mm == True:
        save_ext = 'dat'
    datadict = UD.load_data_dict(cur_dsname, classify_by_subcategory = cls_subcat, tomlfile_str = tomlfile_str)
    cur_df = datadict['df']
    cur_label_arr = datadict['label_arr']
    num_classes =  datadict['num_classes']
    cur_label_col = datadict['label_col']
    idxs = UD.get_train_test_subsets(cur_label_arr, train_on_middle = cur_tom, train_pct = 0.7, test_subpct = 0.5, seed = seed)
    train_idxs = idxs['train']
    test_idxs = idxs['test']
    valid_idxs = idxs['valid']
    train_df = cur_df[train_idxs]
    test_df = cur_df[test_idxs]
    valid_df = cur_df[valid_idxs]
    
    cur_data = UD.collate_data_at_idx(cur_df,layer_idx, cur_embtype,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = False, use_64bit = False, device = '')
    
    cur_km = SKC.KMeans(n_clusters = num_classes, random_state=seed, n_init = 'auto').fit(cur_data)
    cur_clustering = cur_km.labels_





