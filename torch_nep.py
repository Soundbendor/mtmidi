import neptune
import util
import os
import neptune.integrations.optuna as NIO
# https://docs.neptune.ai/integrations/optuna/#more-options
# https://docs.neptune.ai/integrations/optuna/#customizing-which-plots-to-log-and-how-often
def init(params, plots_update_freq = 10, log_plot_slice = True, log_plot_contour = True):
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
    run['parameters'] = params
    callback = NIO.NeptuneCallback(run, plots_update_freq = 10, log_plot_slice = log_plot_slice, log_plot_contour=log_plot_contour)
    return run, callback

def tidy(study, run):
    NIO.log_study_metadata(study, run)
    run.stop()


