import util_data as UD
from distutils.util import strtobool
import polars as pl
import argparse
import os

seed = 5

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-et1", "--embedding_type1", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-et2", "--embedding_type2", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-li1", "--layer_idx1", type=int, default=0, help="specifies layer_idx 0-indexed")
    parser.add_argument("-li2", "--layer_idx2", type=int, default=0, help="specifies layer_idx 0-indexed")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-ti", "--test_index", type=int, default=-1, help="pass index >= 0 to specify test dataset")
    
    args = parser.parse_args()
    arg_dict = vars(args)

    cur_embtype1 = arg_dict['embedding_type1']
    cur_embtype2 = arg_dict['embedding_type2']
    layer_idx1 = arg_dict['layer_idx1']
    layer_idx2 = arg_dict['layer_idx2']
    cur_dsname = arg_dict['dataset']
    tomlfile_str = arg_dict['toml_file']
    cls_subcat = arg_dict['classify_by_subcategory']
    cur_tom = arg_dict['train_on_middle']
    cur_mm = arg_dict['memmap']
    test_idx = arg_dict['test_index']
    is_test = test_idx >= 0
    save_ext = 'npy'
    
    # layer capping
    cur_shape1 = UM.get_embedding_shape(cur_embtype)
    cur_shape2 = UM.get_embedding_shape(cur_embtype)
    cur_emblayers1 = cur_shape1[0]
    cur_emblayers2 = cur_shape2[0]
    if layer_idx1 >= cur_emblayers1:
        old_layer_idx1 = layer_idx1
        layer_idx1 = cur_emblayers1 - 1
        print(f'changing layer_idx1 from {old_layer_idx1} to {layer_idx1}')

   if layer_idx2 >= cur_emblayers2:
        old_layer_idx2 = layer_idx2
        layer_idx2 = cur_emblayers2 - 1
        print(f'changing layer_idx from {old_layer_idx2} to {layer_idx2}')


    if cur_mm == True:
        save_ext = 'dat'

    cur_df = None
    cur_label_col = None
    num_classes = None
    cur_data1 = None
    cur_data2 = None

    if is_test == False:
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
        cur_data1 = UD.collate_data_at_idx(cur_df,layer_idx1, cur_embtype1,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = True, use_64bit = False, device = '')
        cur_data2 = UD.collate_data_at_idx(cur_df,layer_idx2, cur_embtype2,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = True, use_64bit = False, device = '')
    else:
        cur_df_path = UM.by_projpath2(subpaths=['test_csv','cluster','kmeans'], make_dir = False)
        cur_df_file = os.path.join(cur_df_path, f'ds-5000_10-{test_idx}.csv')
        cur_df = pl.read_csv(cur_df_file)
        cur_label_col = 'label' 
        #cur_dat_path = UM.by_projpath2(subpaths=['test_acts','cluster','kmeans'], make_dir = False)
        cur_dat_file = f'ds-5000_10-{test_idx}.dat'
        cur_shape = (5000, 10)

        test_act_folder = 'test_acts'
        test_csv_folder = 'test_csv'
        test_dataset = 'cluster'
        cur_embtype = 'kmeans'
        cur_dsname = 'test'
        layer_idx = test_idx
        num_classes = 10
    
        cur_emb = UM.get_embedding_file(cur_embtype, acts_folder = test_act_folder, dataset=test_dataset, fname=cur_dat_file, write = False, use_64bit = False, use_shape = cur_shape)
        cur_data = cur_emb.copy()
    cur_nc = int(cur_clmult * num_classes)
    cur_clustering = None
    


    cur_clustering = cur_algo.labels_
    if cur_cltype == 'optics':
        cur_nc = cur_algo.cluster_hierarchy_.shape[0]
    elif cur_cltype == 'hdbscan':
        cur_nc = cur_algo.centroids_.shape[0]
    elif cur_cltype == 'dbscan':
        cur_nc = cur_algo.components_.shape[0]
    res_folder = UM.by_projpath2(subpaths=['res_cluster',cur_cltype,cur_dsname, cur_embtype, f'layer_idx-{layer_idx}'], make_dir = True)

    get_cluster_metrics(cur_df,cur_label_col, cur_nc,cur_clustering, res_folder)





