import neptune

def init(params):
    nep_tok = None
    with open('nep_api.txt', 'r') as f:
        nep_tok = f.read().strip()
    run = neptune.init_run(
        project="Soundbendor/SynTheoryPlus",
        api_token=nep_tok,
    )  # your credentials
    run['parameters'] = params
    return run



