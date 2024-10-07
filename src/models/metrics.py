import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import binary_auroc, binary_average_precision


class MultiLabelBinaryEval(Metric):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.logits = {i: [] for i in range(self.num_classes)}
        self.target = {i: [] for i in range(self.num_classes)}

    def update(self, logits, target):
        with torch.no_grad():
            for i in range(self.num_classes):

                self.logits[i].append(logits[:,i])
                self.target[i].append(target[:, i])

    def compute(self):
        with torch.no_grad():

            logits = [torch.concat(self.logits[i]) for i in range(self.num_classes)]
            targets = [torch.concat(self.target[i]) for i in range(self.num_classes)]


            mAP = [binary_average_precision(logits[i], targets[i].int()) for i in range(self.num_classes)]
            auc_roc = [binary_auroc(logits[i], targets[i].int()) for i in range(self.num_classes)]

            return dict(
                mAP=sum(mAP) / len(mAP),
                auc_roc=sum(auc_roc) / len(auc_roc),
            )
    

    def reset(self) -> None:
        super().reset()
        self.logits = {i: [] for i in range(self.num_classes)}
        self.target = {i: [] for i in range(self.num_classes)}

    def _compute_f1(self, logits, target, threshold=0.4):
        tp = torch.count_nonzero((logits > threshold)  & (target == 1))
        fp = torch.count_nonzero((logits > threshold)  & (target == 0))
        tn = torch.count_nonzero((logits <= threshold) & (target == 0))
        fn = torch.count_nonzero((logits <= threshold) & (target == 1))

        precision_p = tp / (tp + fp)
        recall_p = tp / (tp + fn)
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p)
        f1_p = torch.nan_to_num(f1_p, 0.)
        
        precision_n = tn / (tn + fn)
        recall_n = tn / (tn + fp)
        f1_n = 2 * precision_n * recall_n / (precision_n + recall_n)
        return f1_p, (f1_p + f1_n) / 2