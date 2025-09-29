from distutils.util import strtobool
import util_data as UD
import argparse
import util as UM
import os, time, subprocess

def get_layer_idxs(idxarr, emb_type, emb1 = True):
    li_arr = []
    cur_shape = UM.get_embedding_shape(emb_type)
    cur_emblayers = cur_shape[0]
    if len(idxarr) > 1:
        first_idx = max(0, idxarr[0])
        last_idx = idxarr[-1]
        if last_idx >= cur_emblayers:
            old_last_idx = last_idx
            last_idx = cur_emblayers - 1
            e_str = 'embedding_type1'
            if emb1 == False:
                e_str = 'embedding_type2'
            print(f'changing last_idx for {e_str} from {old_last_idx} to {last_idx}')
        li_arr = list(range(first_idx, last_idx+1))
    else:
        cur_idx = max(0,idxarr[0])
        if cur_idx >= cur_emblayers:
            old_cur_idx = cur_idx
            cur_idx = cur_emblayers - 1
            e_str = 'embedding_type1'
            if emb1 == False:
                e_str = 'embedding_type2'
            print(f'changing cur_idx for {e_str} from {old_cur_idx} to {cur_idx}')
        li_arr = [cur_idx]
    return li_arr
    


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-et1", "--embedding_types1", nargs="+", type=str, default=["jukebox"], help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-et2", "--embedding_types2", nargs="+", type=str, default=["jukebox"], help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-li1", "--layer_idxs1", nargs='+', type=int, default=[0], help="layer_idxs of embedding files 0-indexed (if multiple, defines start and end indices)")
    parser.add_argument("-li2", "--layer_idxs2", nargs='+', type=int, default=[0], help="layer_idxs of embedding files 0-indexed (if multiple, defines start and end indices)")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-tf", "--toml_file", type=str, default="", help="toml file in toml directory with exclude category listing vals to exclude by col, amongst other settings")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-ram", "--ram_mem", type=int, default=40, help="ram in gigs")
    parser.add_argument("-gpu", "--gpus", type=int, default=1, help="num of gpus to use")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    


    args = parser.parse_args()
    arg_dict = vars(args)
    
    root_path = UM.by_projpath()
    script_dir = UM.by_projpath(subpath='cka_sh', make_dir = True)
    cka_path = os.path.join(root_path, 'cka_test.py')
    script_idx = 0

    dataset = args.dataset
    ds_abbrev = UM.dataset_abbrev[dataset]

    start_time = str(int(time.time() * 1000))

    for embedding_type1 in args.embedding_types1:
        li_arr1 = get_layer_idxs(args.layer_idxs1, embedding_type1, emb1 = True)
        for embedding_type2 in args.embedding_types2:
            same_emb_type = embedding_type1.strip() == embedding_type2.strip()
            emb_abbrev1 = UM.model_abbrev[embedding_type1]
            emb_abbrev2 = UM.model_abbrev[embedding_type2]
            li_arr2 = get_layer_idxs(args.layer_idxs2, embedding_type2, emb1 = False)
     
            for layer_idx1 in li_arr1:
                for layer_idx2 in li_arr2:
                    same_li = layer_idx1 == layer_idx2
                    #if not (same_emb_type and same_li):
                    if True:
                        slurm_strarr = ["#!/bin/bash", f"#SBATCH -p {args.partition}",f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", "#SBATCH -t 1-00:00:00", f"#SBATCH --job-name={ds_abbrev}_{emb_abbrev1}{emb_abbrev2}cka", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/slurm_out/{ds_abbrev}{emb_abbrev1}{emb_abbrev2}cka-%j.out", ""]
                        p_str = f"python {cka_path} -ds {dataset} -bs {args.batch_size} -et1 {embedding_type1} -et2 {embedding_type2} -li1 {layer_idx1} -li2 {layer_idx2}"
                        if dataset == "polyrhythms" and len(args.toml_file) > 0:
                            p_str = p_str + f" -tf {args.toml_file}"
                        slurm_strarr.append(p_str)
                        script_fname = f"{start_time}_{ds_abbrev}-{emb_abbrev1}-{emb_abbrev2}-{layer_idx1}_{layer_idx2}.sh"
                        script_idx += 1
                        script_path = os.path.join(script_dir, script_fname)
                        script_str = "\n".join(slurm_strarr)
                        print(f"===== {dataset} | {embedding_type1} | {embedding_type2} | LI {layer_idx1}-{layer_idx2} =====")
                        print(f"Creating {script_fname}")
                        with open(script_path, 'w') as f:
                            f.write(script_str)
                        subprocess.run(f"chmod u+x {script_path}", shell=True)
                        if args.debug == False:
                            print(f"Running {script_fname}")
                            subprocess.run(f"sbatch -W {script_path}", shell=True)



