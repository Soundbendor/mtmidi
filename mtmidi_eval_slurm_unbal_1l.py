from distutils.util import strtobool
import util_data as UD
import argparse
import util as UM
import os, time, subprocess


# prefixes for oct run of experiments
def get_prefix(embedding):
    ret_pfix = 0
    if embedding in ['mg_small_h', 'mg_med_h', 'mg_large_h', 'jukebox']:
        ret_pfix = 9155
    else:
        ret_pfix = 9133
    return ret_pfix

        


if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-et", "--embedding_types", nargs="+", type=str, default=["jukebox", "mg_small_h", "mg_med_h", "mg_large_h", "mg_audio", "baseline_mel", "baseline_mfcc", "baseline_chroma", "baseline_concat"], help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-cbs", "--classify_by_subcategory", type=strtobool, default=False, help="classify by subcategory for dynamics, by progression for chord progression datasets")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="hacky way of syntax debugging")
    parser.add_argument("-ram", "--ram_mem", type=int, default=40, help="ram in gigs")
    parser.add_argument("-gpu", "--gpus", type=int, default=1, help="num of gpus to use")
    parser.add_argument("-sh", "--on_share", type=strtobool, default=False, help="on share")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="load embeddings as memmap, else npy")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    


    args = parser.parse_args()
    arg_dict = vars(args)
   
    emblen = len(args.embedding_types)

    root_path = UM.by_projpath()
    script_dir = UM.by_projpath(subpath='eval_sh', make_dir = True)
    py_path = os.path.join(root_path, 'probing.py')
    script_idx = 0

    dataset = 'dynamics'
    ds_abbrev = UM.dataset_abbrev[dataset]

    start_time = str(int(time.time() * 1000))
    batch_size = 64
    cur_cbs = args.classify_by_subcategory
    cur_memmap = args.memmap

    
    for embedding_type in args.embedding_types:
        emb_abbrev = UM.model_abbrev[embedding_type]
        cur_prefix = get_prefix(embedding_type)
     
        slurm_strarr1 = ["#!/bin/bash"]
        slurm_strarr2 = [f"#SBATCH -p {args.partition}"]
        if args.partition != 'preempt':
            if args.partition != 'soundbendor':
                slurm_strarr2 = ['#SBATCH -A eecs', f"#SBATCH -p {args.partition}"]
            else:
                slurm_strarr2 = ['#SBATCH -A soundbendor', f"#SBATCH -p {args.partition}"]
        slurm_strarr3 = [f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", "#SBATCH -t 1-00:00:00", f"#SBATCH --job-name={ds_abbrev}_{emb_abbrev}eval", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/slurm_out/{ds_abbrev}{emb_abbrev}eval-%j.out", ""]
        slurm_strarr = slurm_strarr1 + slurm_strarr2 + slurm_strarr3
        p_str = f"python {py_path} -ds {dataset} -ev True -et {embedding_type} -cbs {cur_cbs} -m {cur_memmap} -pf {cur_prefix} -sh {args.on_share} -nh True"
        slurm_strarr.append(p_str)
        script_fname = f"{start_time}_{ds_abbrev}-{emb_abbrev}.sh"
        script_idx += 1
        script_path = os.path.join(script_dir, script_fname)
        script_str = "\n".join(slurm_strarr)
        print(f"===== EVAL | {dataset} | {embedding_type} =====")
        print(f"Creating {script_fname}")
        with open(script_path, 'w') as f:
            f.write(script_str)
        subprocess.run(f"chmod u+x {script_path}", shell=True)
        if args.debug == False:
            print(f"Running {script_fname}")
            subprocess.run(f"sbatch -W {script_path}", shell=True)



