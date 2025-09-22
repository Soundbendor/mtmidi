import polars as pl
import util as UM
import os,json
import sqlite3

db_folder = UM.by_projpath(subpath='db', make_dir = False)

study_tables = ['alembic_version', 'trial_params', 'studies','trial_system_attributes', 'study_directions', 'trial_user_attributes','study_system_attributes','trial_values','study_user_attributes','trials','trial_heartbeats','version_info','trial_intermediate_values']

def get_dbconn(cur_ds, cur_embtype, prefix=5):
    _fname = f"{prefix}-{cur_ds}-{cur_embtype}.db"
    #_fname = f"{prefix}-{cur_ds}-{cur_embtype}"
    dbpath = os.path.join(db_folder,_fname)
    #dburi = f"sqlite:///{dbpath}"
    #dburi = f"sqlite:///db/{_fname}"
    #print(dburi)
    #return sqlite3.connect(r'{}'.format(dburi))
    return sqlite3.connect(dbpath)

def get_trial_id_from_number(conn,number):
    tv = pl.read_database(query = f'select trial_id from trials where number={number}', connection=conn)
    return tv['trial_id'][0]

def get_number_from_trial_id(conn,trial_id):
    tv = pl.read_database(query = f'select number from trials where trial_id={trial_id}', connection=conn)
    return tv['number'][0]



def get_best_trial_id_val(conn):
    tv = pl.read_database(query = 'select trial_id,value from trial_values', connection=conn)
    best_row = tv[tv.select(pl.col('value').arg_max())[0,0]]
    return best_row

def get_best_trial_params(conn):
    best_row = get_best_trial_id_val(conn)
    best_id = best_row['trial_id'][0]
    best_trial_number = get_number_from_trial_id(conn, best_id)
    best_val = best_row['value'][0]
    best_params = pl.read_database(query = f'select param_name,param_value,distribution_json from trial_params where trial_id={best_id}', connection=conn)
    best_dict = {'trial_id': best_id, 'trial_number': best_trial_number, 'value': best_val}
    return (best_params, best_dict)

def close_dbconn(conn):
    conn.close()

def parse_best_params(best_params_df):
    param_dict = {x['param_name']: {'_value': x['param_value'], 'dict': json.loads(x['distribution_json'])} for x in best_params_df.to_dicts()}
    for param,pdict in param_dict.items():
        cur_dist = pdict['dict']['name']
        if cur_dist == 'CategoricalDistribution':
            cur_choices = pdict['dict']['attributes']['choices']
            choice_idx = int(pdict['_value'])
            param_dict[param]['value'] = cur_choices[choice_idx]
        elif cur_dist == "IntDistribution":
            param_dict[param]['value'] = int(pdict["_value"])
        else:
            param_dict[param]['value'] = pdict["_value"]
    return param_dict

def get_study_attr(conn):
    study_attr = pl.read_database(query = f'select * from study_user_attributes', connection=conn)
    ret_dict = {x['key']: json.loads(x['value_json']) for x in study_attr.to_dicts()}
    return ret_dict

def get_best_params(cur_ds, cur_embtype, prefix=5):
    conn = get_dbconn(cur_ds, cur_embtype, prefix=prefix)
    (best_params, best_dict) = get_best_trial_params(conn)
    param_dict = parse_best_params(best_params)
    attr_dict = get_study_attr(conn)
    close_dbconn(conn)
    #param_dict['best_value'] = best_val
    return (param_dict, best_dict, attr_dict)


