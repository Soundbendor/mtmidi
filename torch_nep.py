import neptune
import util
import os
def init(params):
    nep_tok = None
    nep_path = os.path.join(util.script_dir, 'nep_api.txt')
    with open(nep_path, 'r') as f:
        nep_tok = f.read().strip()
    run = neptune.init_run(
        project="Soundbendor/SynTheoryPlus",
        api_token=nep_tok,
    )  # your credentials
    run['parameters'] = params
    return run



