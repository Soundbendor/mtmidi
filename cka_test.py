from CKA import CKA
import util_data as UD
from distutils.util import strtobool
import polars as pl
import argparse
import os

seed = 5

model_array = ['baseline_mel', 'baseline_chroma', 'baseline_mfcc', 'baseline_concat', 'mg_audio', 'mg_small_h', 'mg_small_at', 'mg_med_h', 'mg_med_at', 'mg_large_h', 'mg_large_at', 'jukebox']
model_to_idx = {k:v for (k,v) in enumerate(model_array)}
idx_to_model = {v:k for (k,v) in model_to_idx.items()}

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
    save_ext = 'npy'
  
    to_order = [(model_to_idx[cur_embtype1], cur_embtype1, layer_idx1), (model_to_idx[cur_embtype2], cur_embtype2, layer_idx2)]
    to_order.sort()

    # cuda stuff
    device ='cpu'
    if torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)

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
    cur_data1 = UD.collate_data_at_idx(cur_df,layer_idx1, cur_embtype1,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = True, use_64bit = False, device = device)
    cur_data2 = UD.collate_data_at_idx(cur_df,layer_idx2, cur_embtype2,is_memmap = cur_mm, acts_folder = 'acts', dataset = cur_dsname, to_torch = True, use_64bit = False, device = device)

    cur_emb = UM.get_embedding_file(cur_embtype, acts_folder = test_act_folder, dataset=test_dataset, fname=cur_dat_file, write = False, use_64bit = False, use_shape = cur_shape)
    cur_data = cur_emb.copy()
    

    embtype_str = f'{to_order[0][1]}-{to_order[1][1]}'
    layer_idx_str = f'layer_idx-{to_order[0][2]}-{to_order[1][2]}'
    res_folder = UM.by_projpath2(subpaths=['res_cka_linear',cur_dsname, embtype_str], make_dir = True)
    out_file = os.path.join(res_folder, f'{layer_idx_str}.txt')
    cur_cka = CKA(kernel_type='linear', device=device)
    cur_cka.update(cur_data1, cur_data2)
    cur_res = cur_cka.get_value()
    with open(out_file, 'w') as f:
        f.write(f'{cur_res}')







