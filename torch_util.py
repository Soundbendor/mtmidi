from torcheval import metrics as TM
import neptune

nep_tok = None
with open('nep_api.txt', 'r') as f:
    nep_tok = f.read().strip()

run = neptune.init_run(
    project="Soundbendor/SynTheoryPlus",
    api_token=nep_tok,
)  # your credentials

not_printable = ["confmat", "multilabel"]
ignorekeys = ["multilabel"]
metric_keys_mc = ["acc1_micro", "acc1_macro","acc3_micro", "acc3_macro", "confmat", "avgprec", "f1_micro", "f1_macro", "prec_micro", "prec_macro", "recall_micro", "auroc"]

csvable_mc = list(set(metric_keys_mc).difference(set(not_printable)))

def metric_creator(num_classes=10, device='cpu'):
    mdict = {}
    mdict["acc1_micro"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=1, device= device)
    mdict["acc1_macro"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=1, device=device)
    mdict["acc3_micro"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=3,device=device)
    mdict["acc3_macro"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=3,device=device)
    mdict["confmat"] = TM.MulticlassConfusionMatrix(num_classes=num_classes, device=device)
    mdict["avgprec"] = TM.MulticlassAUPRC(num_classes=num_classes, device=device)
    mdict["f1_micro"] = TM.MulticlassF1Score(num_classes=num_classes, average="micro", device = device)
    mdict["f1_macro"] = TM.MulticlassF1Score(num_classes=num_classes, average="macro", device = device)
    mdict["prec_micro"] = TM.MulticlassPrecision(num_classes=num_classes, average="micro", device=device)
    mdict["prec_macro"] = TM.MulticlassPrecision(num_classes=num_classes, average="macro", device=device)
    mdict["recall_micro"] = TM.MulticlassRecall(num_classes=num_classes, average="micro", device = device)
    #mdict["macro_recall"] = TM.MulticlassRecall(num_classes=num_classes, average="macro")
    mdict["auroc"] = TM.MulticlassAUROC(num_classes=num_classes, device=device)
