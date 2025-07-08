import sklearn.datasets as SKD
import polars as pl
import numpy as np
from distutils.util import strtobool
import argparse
import os
import util as UM

test_act_folder = 'test_acts'
test_csv_folder = 'test_csv'
test_dataset = 'cluster'
seed = 5
def generate_kmeans(n_samples, n_clusters, n_features, ds_idx = 0):
    cur_std = float(ds_idx + 1)
    cur_x, cur_y, cur_c = SKD.make_blobs(n_samples = n_samples,n_features = n_features, centers = n_clusters, shuffle=True,return_centers = True, cluster_std = cur_std, random_state = seed)
    cur_shape = cur_x.shape
    cur_fname = f"ds-{n_samples}_{n_features}-{ds_idx}"
    cur_emb_type = 'kmeans'
    cur_emb = UM.get_embedding_file(cur_emb_type, acts_folder = test_act_folder, dataset=test_dataset, fname=f'{cur_fname}.dat', write = True, use_64bit = False, use_shape = cur_shape)
    cur_emb[:] = cur_x[:]
    df_data = {'idx': np.arange(n_samples), 'label': [f'Class {y}' for y in cur_y], 'label_idx': cur_y}
    data_df = pl.DataFrame(df_data, schema=[('idx', pl.Int64), ('label', pl.String), ('label_idx', pl.Int64)])

    csv_dir_arr = [test_csv_folder, test_dataset, cur_emb_type]
    csv_path = UM.by_projpath2(subpaths=csv_dir_arr,make_dir = True)
    csv_fname = f'{cur_fname}.csv'
    csv_data_path = os.path.join(csv_path, csv_fname)
    data_df.write_csv(csv_data_path, separator=',')
if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-di', '--dataset_index', type=int, default = 0, help="index of dataset (if creating multiple)")
    parser.add_argument("-et", "--embedding_type", type=str, default="kmeans", help="kmeans")
    parser.add_argument("-ns", "--num_samples", type=int, default=5000, help="number or samples to generate")
    parser.add_argument("-nc", "--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("-nf", "--num_features", type=int, default=10, help="number of features")
    
    args = parser.parse_args()
    arg_dict = vars(args)
    cur_et = arg_dict['embedding_type']
    cur_ns = arg_dict['num_samples']
    cur_nc = arg_dict['num_classes']
    cur_nf = arg_dict['num_features']
    cur_di = arg_dict['dataset_index']
    if cur_et == 'kmeans':
        generate_kmeans(cur_ns, cur_nc, cur_nf, ds_idx = cur_di)

    
