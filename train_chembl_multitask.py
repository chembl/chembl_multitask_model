from onnxruntime.quantization import quantize_dynamic
import onnx
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as D
import pytorch_lightning as pl
import tables as tb
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from collections import Counter
import json


CHEMBL_VERSION = 34
PATH = "."
DATA_FILE = f"mt_data_{CHEMBL_VERSION}.h5"
N_WORKERS = 6  # prefetches data in parallel to have batches ready for traning
BATCH_SIZE = 32  # https://twitter.com/ylecun/status/989610208497360896
LR = 4  # Learning rate. Big value because of the way we are weighting the targets
FP_SIZE = 1024


# PyTorch Dataset that reads batches from a PyTables file
class ChEMBLDataset(D.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with tb.open_file(self.file_path, mode="r") as t_file:
            self.length = t_file.root.fps.shape[0]
            self.n_targets = t_file.root.labels.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with tb.open_file(self.file_path, mode="r") as t_file:
            structure = t_file.root.fps[index]
            labels = t_file.root.labels[index]
        return structure, labels


class ChEMBLMultiTask(pl.LightningModule):
    """
    Architecture borrowed from: https://arxiv.org/abs/1502.02072
    """

    def __init__(self, n_tasks, weights=None):
        super().__init__()
        self.n_tasks = n_tasks

        self.fc1 = nn.Linear(FP_SIZE, 2000)
        self.fc2 = nn.Linear(2000, 100)
        self.dropout = nn.Dropout(0.25)
        self.test_step_outputs = []

        # add an independent output for each task in the output layer
        for n_m in range(n_tasks):
            self.add_module(f"y{n_m}o", nn.Linear(100, 1))
        if weights is not None:
            self.criterion = [
                nn.BCELoss(weight=w) for w in torch.tensor(weights).float()
            ]
        else:
            self.criterion = [nn.BCELoss() for _ in range(n_tasks)]

    def forward(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = F.relu(self.fc2(h1))
        out = [
            torch.sigmoid(getattr(self, f"y{n_m}o")(h2)) for n_m in range(self.n_tasks)
        ]
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        fps, labels = batch
        logits = self.forward(fps)
        loss = torch.tensor(0.0)
        for j, crit in enumerate(self.criterion):
            # mask keeping labeled molecules for each target
            mask = labels[:, j] >= 0.0
            if len(labels[:, j][mask]) != 0:
                # the loss is the sum of all targets loss
                # there are labeled samples for this target in this batch, so we add it's loss
                loss += crit(logits[j][mask], labels[:, j][mask].view(-1, 1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        fps, labels = batch
        out = self.forward(fps)

        y = []
        y_hat = []
        y_hat_proba = []
        for j, out in enumerate(out):
            mask = labels[:, j] >= 0.0
            y_pred = torch.where(out[mask] > 0.5, torch.ones(1), torch.zeros(1)).view(
                1, -1
            )
            if y_pred.shape[1] > 0:
                for l in labels[:, j][mask].long().tolist():
                    y.append(l)
                for p in y_pred.view(-1, 1).tolist():
                    y_hat.append(int(p[0]))
                for p in out[mask].view(-1, 1).tolist():
                    y_hat_proba.append(float(p[0]))

        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = tp / (tp + fp)
        f1 = f1_score(y, y_hat)
        acc = accuracy_score(y, y_hat)
        mcc = matthews_corrcoef(y, y_hat)
        auc = roc_auc_score(y, y_hat_proba)

        metrics = {
            "test_acc": torch.tensor(acc),
            "test_sens": torch.tensor(sens),
            "test_spec": torch.tensor(spec),
            "test_prec": torch.tensor(prec),
            "test_f1": torch.tensor(f1),
            "test_mcc": torch.tensor(mcc),
            "test_auc": torch.tensor(auc),
        }
        self.log_dict(metrics)
        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        sums = Counter()
        counters = Counter()
        for itemset in self.test_step_outputs:
            sums.update(itemset)
            counters.update(itemset.keys())
        metrics = {x: float(sums[x]) / counters[x] for x in sums.keys()}
        return metrics


if __name__ == "__main__":

    # each task loss is weighted inversely proportional to its number of datapoints, borrowed from:
    # from: http://www.bioinf.at/publications/2014/NIPS2014a.pdf
    with tb.open_file(f"{PATH}/{DATA_FILE}", mode="r") as t_file:
        weights = t_file.root.weights[:]

    dataset = ChEMBLDataset(f"{PATH}/{DATA_FILE}")
    indices = list(range(len(dataset)))

    metrics = []
    kfold = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kfold.split(indices):
        train_sampler = D.sampler.SubsetRandomSampler(train_idx)
        test_sampler = D.sampler.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=train_sampler
        )

        test_loader = DataLoader(
            dataset, batch_size=1000, num_workers=N_WORKERS, sampler=test_sampler
        )

        model = ChEMBLMultiTask(len(weights), weights)

        # this shallow model trains quicker in CPU
        trainer = pl.Trainer(max_epochs=3, accelerator="cpu")
        trainer.fit(model, train_dataloaders=train_loader)
        mm = trainer.test(dataloaders=test_loader)
        metrics.append(mm)

    # average folds metrics
    metrics = [item for sublist in metrics for item in sublist]
    sums = Counter()
    counters = Counter()
    for itemset in metrics:
        sums.update(itemset)
        counters.update(itemset.keys())
    performance = {x: float(sums[x]) / counters[x] for x in sums.keys()}
    with open(f"performance_{CHEMBL_VERSION}.json", "w") as f:
        json.dump(performance, f)

    # Train the model with the whole dataset and export to ONNX format
    final_train_sampler = D.sampler.SubsetRandomSampler(indices)
    final_train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        sampler=final_train_sampler,
    )

    model = ChEMBLMultiTask(len(weights), weights)

    # this shallow model trains quicker in CPU
    trainer = pl.Trainer(max_epochs=3, accelerator="cpu")
    trainer.fit(model, train_dataloaders=final_train_loader)

    with tb.open_file(f"mt_data_{CHEMBL_VERSION}.h5", mode="r") as t_file:
        output_names = t_file.root.target_chembl_ids[:]

    model.to_onnx(
        f"./chembl_{CHEMBL_VERSION}_multitask.onnx",
        torch.ones(FP_SIZE),
        export_params=True,
        input_names=["input"],
        output_names=output_names,
    )

    model_fp32 = f"./chembl_{CHEMBL_VERSION}_multitask.onnx"
    model_quant = f"./chembl_{CHEMBL_VERSION}_multitask_q8.onnx"
    quantized_model = quantize_dynamic(model_fp32, model_quant)
