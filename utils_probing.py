import sklearn.metrics as SKM
import polyrhythms as PL
import matplotlib.pyplot as plt
import util as UM
import neptune
import tempi as TP
import os



res_dir = UM.by_projpath("res", make_dir = True)

def init(class_binsize):
    TP.init(class_binsize)


def get_classification_metrics(truths, preds, save_confmat=True):
    acc = SKM.accuracy_score(truths, preds)
    f1_macro = SKM.f1_score(truths, preds, average='macro')
    f1_micro = SKM.f1_score(truths, preds, average='micro')
    class_truths = [PL.rev_polystr_to_idx[x] for x in truths]
    class_preds = [PL.rev_polystr_to_idx[x] for x in preds]
    cm = None
    if save_confat == True:
        cm = SKM.confusion_matrix(class_truths, class_preds)
        cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=PL.class_arr)
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        _cmd.plot(ax=ax)
        timestamp = int(time.time()*1000)

        cm_fname = f'{timestamp}-cm.png' 
        cm_path = os.path.join(res_dir, cm_fname)
        plt.savefig(cm_path)
        plt.clf()
    d = {'accuracy_score': acc, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'confmat': cm}
    return d

def get_regression_metrics(truths, truth_labels, preds, pred_labels, dataset, held_out_classes = False, save_confmat = True):
    mse = SKM.mean_squared_error(truths, preds)
    r2 = SKM.r2_score(truths, preds)
    mae = SKM.mean_absolute_error(truths, preds)
    ev = SKM.explained_variance_score(truths, preds)
    medae = SKM.median_absolute_error(truths, preds)
    maxerr = SKM.max_error(truths, preds)
    mape = SKM.mean_absolute_percentage_error(truths, preds)
    rmse = SKM.root_mean_squared_error(truths, preds)
    d2ae = SKM.d2_absolute_error_score(truths, preds)
    acc = SKM.accuracy_score(truth_labels, pred_labels)
    f1_macro = SKM.f1_score(truth_labels, pred_labels, average='macro')
    f1_micro = SKM.f1_score(truth_labels, pred_labels, average='micro')

    
    class_truths = None
    class_preds = None
    cm = None
    if save_confmat == True:
        if dataset == 'polyrhythms':
            class_truths = [PL.reg_rev_polystr_to_idx[x] for x in truth_labels]
            class_preds = [PL.reg_rev_polystr_to_idx[x] for x in pred_labels]
        elif dataset == 'tempi':
            class_truths = [TP.rev_classdict[x] for x in truth_labels]
            class_preds = [TP.rev_classdict[x] for x in pred_labels]
        cm = SKM.confusion_matrix(class_truths, class_preds)
        cmd = None
        if held_out_classes == True or dataset != 'polyrhythms':
            all_labels = set(class_truths).union(set(class_preds))
            class_arr2 = None
            if dataset == 'polyrhythms':
                class_arr2 = [x for x in PL.reg_class_arr if x in all_labels]
            else:
                class_arr2 = [x for x in classset_aug if x in all_labels]
            _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=class_arr2)
        else:
            _cmd = SKM.ConfusionMatrixDisplay(confusion_matrix=_cm, display_labels=PL.reg_class_arr)
        _cmd.plot()
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        _cmd.plot(ax=ax)
        timestamp = int(time.time()*1000)
        cm_fname = f'{timestamp}-cm.png' 
        cm_path = os.path.join(res_dir, cm_fname)
        plt.savefig(cm_path)
        plt.clf()

    d = {'mean_squared_error': mse, "r2_score": r2, "mean_absolute_error": mae,
         "explained_variance_score": ev, "median_absolute_error": medae,
         "max_error": eaxerr, "mean_absolute_percentage_error": mape,
         "root_mean_squared_error": rmse, "d2_absolute_error_score": d2ae,
         "accuracy_score": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}
    return d


def init(params):
    nep_tok = None
    nep_path = os.path.join(util.script_dir, 'nep_api.txt')
    with open(nep_path, 'r') as f:
        nep_tok = f.read().strip()
    run = neptune.init_run(
        project="Soundbendor/SynTheoryPlus",
        api_token=nep_tok,
    )  
    run['parameters'] = params
    return run



