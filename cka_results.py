from distutils.util import strtobool
import util_data as UD
import argparse
import cka_util as CU
import util as UM
import os, time, subprocess
import numpy as np

import matplotlib.pyplot as plt

clrmap = 'pink'
if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--datasets", nargs="+", type=str, default=["polyrhythms"], help="dataset")
    parser.add_argument("-et1", "--embedding_types1", nargs="+", type=str, default=["jukebox"], help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-et2", "--embedding_types2", nargs="+", type=str, default=["jukebox"], help="mg_{small/med/large}_{h/at} / mg_audio / jukebox")
    parser.add_argument("-pa", "--print_array", type=strtobool, default=False, help="print array")
    


    args = parser.parse_args()
    arg_dict = vars(args)
    
    root_path = UM.by_projpath()
    script_dir = UM.by_projpath(subpath='cka_sh', make_dir = True)
    cka_path = os.path.join(root_path, 'cka_test.py')
    script_idx = 0


    for dataset in args.datasets:

        ds_abbrev = UM.dataset_abbrev[dataset]
        ds_folder = UM.by_projpath2(subpaths=['res_cka_linear', dataset], make_dir = False)
        if os.path.isdir(ds_folder) == True:
            for et1 in args.embedding_types1:
                for et2 in args.embedding_types2:
                    et_tup, et_str, _ = CU.get_embtype_layeridx_str(et1, 0, et2, 0)
                    et_path = os.path.join(ds_folder, et_str)
                    if os.path.isdir(et_path) == True:
                        has_all_embeddings = True


                        # smaller model should be x axis
                        
                        ## num layers in sorted order
                        _et1 = et_tup[0]
                        _et2 = et_tup[1]
                        _etn1 = CU.get_num_layers(et_tup[0])
                        _etn2 = CU.get_num_layers(et_tup[1])
                        res_arr = np.zeros((_etn2, _etn1), dtype=np.float64)
                        for n1 in range(_etn1):
                            for n2 in range(_etn2):
                                _, _, li_str = CU.get_embtype_layeridx_str(et1, n1, et2, n2)
                                li_path = CU.get_layer_idx_file(et_path, li_str)
                                if os.path.exists(li_path) == True:
                                    cur_num = 0.
                                    with open(li_path, 'r') as f:
                                        try:
                                            cur_num = float(f.read())
                                        except:
                                            print(f'{li_path} not a number')
                                            has_all_embeddings = False

                                    # smaller model goes along columns
                                    res_arr[n2][n1] = cur_num
                                else: 
                                    print(f'{li_path} does not exist')
                                    has_all_embeddings = False

                                if has_all_embeddings == False:
                                    break
                            if has_all_embeddings == False:
                                break
                        if has_all_embeddings == True:
                            if args.print_array == True:
                                print(res_arr)
                            
                            ax = plt.gca()
                            im = ax.imshow(res_arr, norm = 'logit', origin='lower', cmap=clrmap)
                            plt.colorbar(im, label='CKA')
                            #im = ax.imshow(res_arr, norm = None, vmin = np.min(res_arr), vmax = np.max(res_arr), origin='lower', cmap=clrmap)
                            plt.suptitle(f'{dataset}: {_et1} x {_et2}')
                            ax.set_xlabel(_et1)
                            ax.set_ylabel(_et2)
                            
                            graph_folder = UM.by_projpath2(subpaths=['res_cka_linear', 'graph'], make_dir = True)
                            fname = f'ckalin-{ds_abbrev}-{_et1}_{_et2}.png'
                            print(f'writing {fname}')
                            graph_path = os.path.join(graph_folder, fname)
                            plt.tight_layout()
                            plt.savefig(graph_path)
                            plt.clf()

                        




