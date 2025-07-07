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
def generate_kmeans(n_samples, n_clusters, n_features, ds_idx = 0):
    cur_x, cur_y, cur_c = SKD.make_blobs(n_samples = n_samples,n_features = n_features, centers = n_clusters, shuffle=True,return_centers = True)
    cur_shape = cur_x.shape
    cur_fname = f"ds-{n_samples}_{n_features}-{ds_idx}"
    cur_emb_type = 'kmeans'
    cur_emb = UM.get_embedding_file(cur_emb_type, acts_folder = test_act_folder, dataset=test_dataset, fname=cur_fname, write = True, use_64bit = False, use_shape = cur_shape)
    df_data = {'idx': np.arange(n_samples), 'label': cur_y}
    data_df = pl.DataFrame(df_data, schema=[('idx', pl.Int64), ('label', pl.Int64)])

    csv_dir_arr = [test_csv_folder, test_dataset, cur_emb_type]
    csv_path = UM.by_projpath2(subpaths=csv_dir_arr,make_dir = True)
    csv_fname = f'{cur_fname}-data.csv'
    centers_fname = f'{cur_fname}-centers.csv'
    csv_data_path = os.path.join(csv_path, csv_fname)
    csv_centers_path = os.path.join(csv_path, centers_fname)
    data_df.write_csv(csv_data_path, separator=',')
    df_centers = {'label': np.arange(n_clusters)}
    feat_cols = ['f_{i}' for i in range(n_features)]

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-et", "--embedding_type", type=str, default="jukebox", help="kmeans")
    parser.add_argument("-ns", "--num_samples", type=int, default=5000, help="number or samples to generate")
    parser.add_argument("-nc", "--num_classes", type=int, default=5, help="number of classes")
    parser.add_argument("-nf", "--num_classes", type=int, default=10, help="number of features")
    
    args = parser.parse_args()
    arg_dict = vars(args)

    
