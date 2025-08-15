import util as UM
import os

model_array = ['baseline_mel', 'baseline_chroma', 'baseline_mfcc', 'baseline_concat', 'mg_audio', 'mg_small_h', 'mg_small_at', 'mg_med_h', 'mg_med_at', 'mg_large_h', 'mg_large_at', 'jukebox']
model_to_idx = {k:v for (v,k) in enumerate(model_array)}
idx_to_model = {v:k for (k,v) in model_to_idx.items()}

def get_num_layers(emb_type):
    cur_shape = UM.get_embedding_shape(emb_type)
    cur_emblayers = cur_shape[0]
    return cur_emblayers
    
# imposing some sort of ordering of embedding types for folder/file naming/organizational purposes
def get_embtype_layeridx_str(cur_embtype1, layer_idx1, cur_embtype2, layer_idx2):
    to_order = [(model_to_idx[cur_embtype1], cur_embtype1, layer_idx1), (model_to_idx[cur_embtype2], cur_embtype2, layer_idx2)]
    to_order.sort()

    embtype_str = f'{to_order[0][1]}-{to_order[1][1]}'
    layer_idx_str = f'layer_idx-{to_order[0][2]}-{to_order[1][2]}'
    
    return (cur_embtype1, cur_embtype2), embtype_str, layer_idx_str

def get_results_folder(ds_name, embtype_str, make_dir = True):
    res_folder = UM.by_projpath2(subpaths=['res_cka_linear', ds_name, embtype_str], make_dir = True)
    return res_folder


def get_layer_idx_file(res_folder, li_str):
    out_file = os.path.join(res_folder, f'{li_str}.txt')
    return out_file
