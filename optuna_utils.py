import optuna
import os
import time
import pickle
def create_or_load_study(study_base_name, sampler = None, maximize = True, num_trials=3000, prefix=-1, script_dir = os.path.dirname(__file__), sampler_dir = 'samplers', db_dir = 'db'): 
    cur_prefix = int(time.time() * 1000)
    if prefix > 0:
        cur_prefix = prefix
    study_name = f"{cur_prefix}-{study_base_name}-{num_trials}"

    sampler_path = os.path.join(script_dir, sampler_dir)
    rdb_path = os.path.join(script_dir, db_dir)
    if os.path.exists(sampler_path) == False:
        os.makedirs(sampler_path)

    if os.path.exists(rdb_path) == False:
        os.makedirs(rdb_path)
    sampler_file = os.path.join(sampler_path, f'{study_name}.pkl')
    rdb_file = os.path.join(rdb_path, f'{study_name}.db')
    resuming = False
    cur_sampler = None
    if os.path.exists(rdb_file) == True and os.path.exists(sampler_file) == True:
        resuming = True
        cur_sampler = pickle.load(open(sampler_file, 'rb'))
    else:
        cur_sampler = sampler
    rdb_string_url = "sqlite:///" + rdb_file
    direction_string = 'maximize'
    study = None
    ret_dict = {}
    ret_dict['study_name'] = study_name
    ret_dict['sampler_fpath'] = sampler_file
    ret_dict['rdb_fpath'] = rdb_file
    ret_dict['prefix'] = cur_prefix
    ret_dict['resuming'] = resuming
    if maximize == False:
        direction_string = 'minimize'
    if cur_sampler == None:
        study = optuna.create_study(study_name=study_name, storage=rdb_string_url, direction=direction_string, load_if_exists = True)
    else:
        study = optuna.create_study(sampler=cur_sampler, study_name=study_name, storage=rdb_string_url, direction=direction_string, load_if_exists=True)
    ret_dict['study'] = study
    return ret_dict
