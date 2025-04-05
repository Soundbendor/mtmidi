import neptune
import util
import os
import neptune.integrations.optuna as NIO
# https://docs.neptune.ai/integrations/optuna/#more-options
# https://docs.neptune.ai/integrations/optuna/#customizing-which-plots-to-log-and-how-often
def init(param_dict=None, plots_update_freq = "never", log_plot_slice = False, log_plot_contour = False):
    nep_tok = None
    nep_path = os.path.join(util.script_dir, 'nep_api.txt')
    with open(nep_path, 'r') as f:
        nep_tok = f.read().strip()
    run = neptune.init_run(
        project="Soundbendor/SynTheoryPlus",
        api_token=nep_tok,
        capture_hardware_metrics = False,
        capture_stderr = False,
        capture_stdout=False,
        capture_traceback=False,
        git_ref=False,
        source_files=[]
    )  # your credentials
    if param_dict != None:
        run['parameters'] = param_dict
    #run['parameters'] = params
    callback = NIO.NeptuneCallback(run, log_all_trials = True, plots_update_freq = "never", log_plot_slice = log_plot_slice, log_plot_contour=log_plot_contour)
    return run, callback

def tidy(study, run):
    NIO.log_study_metadata(study, run)
    run.stop()


