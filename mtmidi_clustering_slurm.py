import numpy as np
from distutils.util import strtobool
import util_data as UD
import polars as pl
import argparse
import sklearn.cluster as SKC
import sklearn.metrics as SKM
import util as UM
import os, time, subprocess


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", nargs='+', type=str, default=["polyrhythms"], help="datasets")
    parser.add_argument("-ct", "--cluster_type", nargs='+', type=str, default=["kmeans"], help="clustering types")
    parser.add_argument("-et", "--embedding_type", nargs='+', type=str, default=["jukebox"], help="multiple from mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-cm", "--cluster_mult", nargs='+', type=float, default=[1.], help="search for num_classes * cluster_mult classes (multiple)")
    parser.add_argument("-li", "--layer_idx", nargs='+', type=int, default=[0], help="layer_idxs of embedding files 0-indexed")
    parser.add_argument("-cls", "--is_classification", type=strtobool, default=True, help="is classification")
    parser.add_argument("-tom", "--train_on_middle", type=strtobool, default=False, help="train on middle")
    parser.add_argument("-rc", "--do_regression_classification", type=strtobool, default=False, help="do regression classification")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-tf", "--toml_file", type=str, default="poly_ex3", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    parser.add_argument("-ti", "--test_index", type=int, default=-1, help="pass index > 0 to specify test dataset")
    parser.add_argument("-mi", "--max_iter", type=int, default=10000, help="maximum number of iterations")
    parser.add_argument("-cpm", "--cpu_mem", type=int, default=5, help="cpu_memory in gigs")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")

    args = parser.parse_args()
    arg_dict = vars(args)
    
    root_path = UM.by_projpath()
    script_dir = UM.by_projpath(subpath='sh', make_dir = True)
    clust_path = os.path.join(root_path, 'clustering.py')
    script_idx = 0

    start_time = str(int(time.time() * 1000))
    for dataset in args.dataset:
        for cluster_type in args.cluster_type:
            for embedding_type in args.embedding_type:
                for cluster_mult in args.cluster_mult:
                    for layer_idx in args.layer_idx:

                        emb_abbrev = UM.model_abbrev[embedding_type]
                        ds_abbrev = UM.dataset_abbrev[dataset]
                        cl_abbrev = UM.cluster_abbrev[cluster_type]
                        slurm_strarr = ["#!/bin/bash", f"#SBATCH -p {args.partition}",f"#SBATCH --mem={args.cpu_mem}G","#SBATCH -t 1-00:00:00", f"#SBATCH --job-name={ds_abbrev}_{emb_abbrev}{cl_abbrev}", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/slurm_out/{ds_abbrev}{emb_abbrev}{cl_abbrev}-%j.out", ""]
                        p_str = f"python {clust_path} -ds {dataset} -et {embedding_type} -ct {cluster_type} -li {layer_idx} -cm {cluster_mult}"
                        if dataset == "polyrhythms":
                            p_str = p_str + f" -tf {args.toml_file}"
                        slurm_strarr.append(p_str)
                        cm_str = str(int(cluster_mult * 1000))
                        script_fname = f"{start_time}_{ds_abbrev}-{cl_abbrev}-{emb_abbrev}-{cm_str}-{layer_idx}.sh"
                        script_idx += 1
                        script_path = os.path.join(script_dir, script_fname)
                        script_str = "\n".join(slurm_strarr)
                        print(f"=== {dataset} | {cluster_type} | {embedding_type} | CM: {cluster_mult} | LI {layer_idx} ===")
                        print(f"Creating {script_path}")
                        with open(script_path, 'w') as f:
                            f.write(script_str)
                        subprocess.run(f"chmod u+x {script_path}", shell=True)
                        if args.debug == False:
                            print(f"Running {script_path}")
                            subprocess.run(f"sbatch -W {script_path}", shell=True)



