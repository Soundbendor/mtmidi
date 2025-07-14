import numpy as np
from distutils.util import strtobool
import util_data as UD
import polars as pl
import argparse
import sklearn.cluster as SKC
import sklearn.metrics as SKM
import util as UM
import os

seed = 5

def get_cluster_metrics(cur_df,cur_labelcol, num_clusters,cur_clustering, out_folder):
    unique_labels = cur_df[cur_labelcol].unique().to_numpy().flatten()
    labeldict = {x:i for (i,x) in enumerate(unique_labels)}
    rev_labeldict = {i:x for (x,i) in labeldict.items()}
    cur_df = cur_df.with_columns(label_idx = pl.col(cur_labelcol).replace_strict(labeldict).cast(int))
    cur_fprefix = f'nc_{num_clusters}'
    # -------------- stuff for purity testing -------------
    #default_props = {k: 0.0 for k in unique_labels}
    cluster_series = pl.Series('cluster', cur_clustering)
    cur_df.insert_column(-1, cluster_series)
    res = [None for x in range(num_clusters)]
    c_idxs = []
    max_labels = []
    max_proportions = []
    for c_idx in range(num_clusters):
        cur_props = cur_df.filter(pl.col('cluster') == c_idx)[cur_labelcol].value_counts(normalize=True,sort=True)
        cur_max = cur_props[0]
        max_label = cur_max[cur_labelcol][0]
        max_prop = cur_max['proportion'][0]
        #cur_res = default_props | cur_props.rows_by_key(key=[cur_labelcol],unique=True)
        c_idxs.append(c_idx)
        max_labels.append(max_label)
        max_proportions.append(max_prop)
        #res[c_idx] = cur_res | {'max_label': max_label, 'max_proportion': max_prop}
        c_fname = os.path.join(out_folder, f'{cur_fprefix}-purity-cluster_{c_idx}-res.csv')
        cur_props.write_csv(c_fname, separator = ",")
    overall_res = {'cluster': c_idxs, 'max_label': max_labels, 'max_proportion': max_proportions}
    overall_df = pl.DataFrame(overall_res, schema=[('cluster', pl.Int64), ('max_label', pl.String), ('max_proportion', pl.Float32)])
    o_fname = os.path.join(out_folder, f'{cur_fprefix}-purity-overall-res.csv')
    overall_df.write_csv(o_fname, separator = ",")

    # other metrics
    use_metrics = {'adj_rand': SKM.adjusted_rand_score,
                   'adj_mi': lambda x,y: SKM.adjusted_mutual_info_score(x,y,average_method='arithmetic'),
                   'hg_score': SKM.homogeneity_score,
                   'complete_score': SKM.completeness_score,
                   'v_meas': lambda x,y: SKM.v_measure_score(x,y,beta=1.0),
                   'fm_score': SKM.fowlkes_mallows_score}
    real_idx = cur_df['label_idx'].to_numpy().flatten()
    calc_metrics = {k:fn(real_idx, cur_clustering) for (k,fn) in use_metrics.items()}
    res_df = pl.from_dict(calc_metrics)
    o_fname2 = os.path.join(out_folder, f'{cur_fprefix}-other-overall-res.csv')
    res_df.write_csv(o_fname2, separator = ",")
    

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ct", "--cluster_type", type=str, default="kmeans", help="clustering type (kmeans, spectral, ward, avg_agg, complete_agg, single_agg, dbscan, hdbscan, optics, birch")
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-cm", "--cluster_mult", type=float, default=1, help="search for num_classes * cluster_mult classes")
    parser.add_argument("-li", "--layer_idx", type=int, default=0, help="specifies layer_idx 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=False, help="train on middle")
    parser.add_argument("-rc", "--do_regression_classification", type=strtobool, default=False, help="do regression classification")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-ti", "--test_index", type=int, default=-1, help="pass index > 0 to specify test dataset")
    parser.add_argument("-mi", "--max_iter", type=int, default=10000, help="maximum number of iterations")
    parser.add_argument("-ms", "--min_samples", type=int, default=5, help="min samples, used for (h)dbscan and optics")
    parser.add_argument("-eps", "--eps", type=float, default=0.5, help="eps, used for dbscan")
    parser.add_argument("-tr", "--threshold", type=float, default=0.5, help="threshold for birch")
    
    args = parser.parse_args()
    arg_dict = vars(args)

    cur_clmult = arg_dict['cluster_mult']
    cur_cltype = arg_dict['cluster_type']
    cur_embtype = arg_dict['embedding_type']
    layer_idx = arg_dict['layer_idx']
    cur_dsname = arg_dict['dataset']
    tomlfile_str = arg_dict['toml_file']
    cls_subcat = arg_dict['classify_by_subcategory']
    cur_tom = arg_dict['train_on_middle']
    cur_mm = arg_dict['memmap']
    cur_mi = arg_dict['max_iter']
    test_idx = arg_dict['test_index']
    cur_minsamp = arg_dict['min_samples']
    is_test = test_idx >= 0
    save_ext = 'npy'
    if cur_mm == True:
        save_ext = 'dat'

    cur_df = None
    cur_label_col = None
    num_classes = None
    cur_data = None

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
        cur_data = UD.collate_data_at_idx(cur_df,layer_idx, cur_embtype,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = False, use_64bit = False, device = '')
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
    
    cur_algo = None

    if cur_cltype not in UM.cluster_types:
        print("incorrect cluster type")
        exit()

    if cur_cltype == 'kmeans':
        cur_algo = SKC.KMeans(n_clusters = cur_nc, random_state=seed, max_iter = cur_mi,  n_init = 'auto').fit(cur_data)
    elif cur_cltype == 'spectral':
        cur_algo = SKC.SpectralClustering(n_clusters = cur_nc, random_state = seed, eigen_solver = 'amg', assign_labels = 'kmeans', n_init = 10).fit(cur_data)
    elif cur_cltype in ['ward', 'avg_agg', 'complete_agg', 'single_agg']:
        agg_str = 'ward'
        if cur_cltype == 'avg_agg':
            agg_str = 'average'
        elif cur_cltype == 'complete_agg':
            agg_str = 'complete'
        elif cur_cltype == 'single_agg':
            agg_str = 'single'
        cur_algo = SKC.AgglomerativeClustering(n_clusters = cur_nc, metric='euclidean',  linkage = agg_str)
    elif cur_cltype in ['dbscan', 'hdbscan']:
        if cur_cltype == 'dbscan':
            cur_algo = SKC.DBSCAN(eps = args.eps, min_samples = args.min_samples, p =2)
        elif cur_cltype == 'hdbscan':
            cur_algo = SKC.HDBSCAN(min_cluster_size = min_samples)
        # set number of clusters to 0 for naming consistency
        cur_nc = 0
    elif cur_cltype == "optics":
        cur_algo = SKC.OPTICS(min_samples = args.min_samples, metric='minkowski', p=2)
    elif  cur_cltype == "birch":
        cur_algo = SKC.Birch(threshold = args.threshold, n_clusters = cur_nc)




    cur_clustering = cur_algo.labels_
    res_folder = UM.by_projpath2(subpaths=['res_cluster',cur_cltype,cur_dsname, cur_embtype, f'layer_idx-{layer_idx}'], make_dir = True)

    get_cluster_metrics(cur_df,cur_label_col, cur_nc,cur_clustering, res_folder)




