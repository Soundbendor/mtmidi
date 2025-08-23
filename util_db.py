import polars as pl
import util as UM
import os
import sqlite3

db_folder = UM.by_projpath(subpath='db', make_dir = False)

study_tables = ['alembic_version', 'trial_params', 'studies','trial_system_attributes', 'study_directions', 'trial_user_attributes','study_system_attributes','trial_values','study_user_attributes','trials','trial_heartbeats','version_info','trial_intermediate_values']

def get_dbconn(cur_ds, cur_embtype, pfix=5):
    _fname = f"{pfix}-{cur_ds}-{cur_embtype}.db"
    #_fname = f"{pfix}-{cur_ds}-{cur_embtype}"
    dbpath = os.path.join(db_folder,_fname)
    #dburi = f"sqlite:///{dbpath}"
    #dburi = f"sqlite:///db/{_fname}"
    #print(dburi)
    #return sqlite3.connect(r'{}'.format(dburi))
    return sqlite3.connect(dbpath)

def get_best_trial_id_val(conn):
    tv = pl.read_database(query = 'select trial_id,value from trial_values', connection=conn)
    best_row = tv[tv.select(pl.col('value').arg_max())[0,0]]
    return best_row

def get_best_trial_params(conn):
    best_row = get_best_trial_id_val(conn)
    best_id = best_row['trial_id'][0]
    best_val = best_row['value'][0]
    best_params = pl.read_database(query = f'select param_name,param_value from trial_params where trial_id={best_id}', connection=conn)
    return (best_params, best_val)

def close_dbconn(conn):
    conn.close()

def get_best_params(cur_ds, cur_embtype, pfix=5):
    conn = get_dbconn(cur_ds, cur_embtype, pfix=pfix)
    (best_params, best_val) = get_best_trial_params(conn)
    close_dbconn(conn)
    param_dict = {x['param_name']: x['param_value'] for x in best_params.to_dicts()}
    #param_dict['best_value'] = best_val
    return (param_dict, best_val)


