import argparse
import os
from onnxruntime.quantization import quantize_dynamic
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
)
from sklearn.model_selection import KFold
from collections import Counter
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a multi-task model on ChEMBL data.")
    # Specify ChEMBL version to use
    parser.add_argument('--chembl_version', type=int, required=True, help="ChEMBL version")
    # Path to the input dataset file
    parser.add_argument('--data_file', type=str, required=True, help="Path to the data file")
    # Define model training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size") # https://twitter.com/ylecun/status/989610208497360896
    parser.add_argument('--lr', type=float, default=4.0, help="Learning rate")  # Big value because of the way we are weighting the targets
    parser.add_argument('--max_epochs', type=int, default=3, help="Maximum number of epochs")
    parser.add_argument('--n_workers', type=int, default=6, help="Number of workers for data loading")
    parser.add_argument('--cv_folds', type=int, default=6, help="Number of K-Fold CV folds (0 to train model with whole dataset)")
    # Directory for saving output files
    parser.add_argument('--output_dir', type=str, default='./', help="Directory to save results")
    return parser.parse_args()

# Dataset class to handle ChEMBL data stored in PyTables format
class ChEMBLDataset(D.Dataset):
    def __init__(self, file_path):
        # Load file and initialize dataset size and number of tasks
        self.file_path = file_path
        with tb.open_file(self.file_path, mode="r") as t_file:
            self.length = t_file.root.fps.shape[0]  # Number of samples
            self.n_targets = t_file.root.labels.shape[1]  # Number of targets/tasks

    def __len__(self):
        # Return dataset size
        return self.length

    def __getitem__(self, index):
        # Fetch fingerprints and labels for a given sample index
        with tb.open_file(self.file_path, mode="r") as t_file:
            structure = t_file.root.fps[index]  # Molecular fingerprints
            labels = t_file.root.labels[index]  # Target labels
        return structure, labels

# PyTorch Lightning module for the multi-task model
class ChEMBLMultiTask(pl.LightningModule):
    """
    Multi-task learning model architecture inspired by: https://arxiv.org/abs/1502.02072
    Supports flexible numbers of tasks with independent outputs for each target.
    """

    def __init__(self, n_tasks, fp_size, lr, weights=None):
        """
        Initialize the multi-task model with independent output layers for each task.

        Args:
            n_tasks (int): Number of prediction tasks.
            fp_size (int): Size of the input fingerprint vector.
            weights (list, optional): Task-specific loss weights to address class imbalance.
        """
        super().__init__()
        self.n_tasks = n_tasks  # Number of tasks to predict
        # Define the layers of the network
        self.fc1 = nn.Linear(fp_size, 2000)  # First fully connected layer
        self.fc2 = nn.Linear(2000, 100)     # Second fully connected layer
        self.dropout = nn.Dropout(0.25)    # Dropout layer for regularization
        self.test_step_outputs = []  # Store test step outputs for post-processing
        self.lr = lr

        # Add an independent output layer for each task
        for n_m in range(n_tasks):
            self.add_module(f"y{n_m}o", nn.Linear(100, 1))  # Output layer for task `n_m`

        # Define loss functions for each task
        if weights is not None:
            # Weighted Binary Cross Entropy Loss for each task
            self.criterion = [
                nn.BCELoss(weight=w) for w in torch.tensor(weights, dtype=torch.float32)
            ]
        else:
            # Unweighted Binary Cross Entropy Loss for each task
            self.criterion = [nn.BCELoss() for _ in range(n_tasks)]

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input feature vector.

        Returns:
            list[Tensor]: List of outputs, one for each task.
        """
        h1 = self.dropout(F.relu(self.fc1(x)))  # First layer with ReLU and dropout
        h2 = F.relu(self.fc2(h1))  # Second layer with ReLU
        # Output layers for all tasks
        out = [
            torch.sigmoid(getattr(self, f"y{n_m}o")(h2)) for n_m in range(self.n_tasks)
        ]
        return out

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            optimizer: Stochastic Gradient Descent optimizer.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)  # Use SGD with learning rate `LR`
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): A tuple containing input fingerprints and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Loss for the current batch.
        """
        fps, labels = batch  # Unpack input data
        logits = self.forward(fps)  # Forward pass
        loss = torch.tensor(0.0)  # Initialize loss accumulator

        # Compute the loss for each task
        for j, crit in enumerate(self.criterion):
            mask = labels[:, j] >= 0.0  # Mask to exclude invalid labels
            if len(labels[:, j][mask]) != 0:  # Skip tasks with no valid labels
                loss += crit(logits[j][mask], labels[:, j][mask].view(-1, 1))  # Compute task loss

        # Log training loss for monitoring
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step, computing predictions and metrics.

        Args:
            batch (tuple): A tuple containing input fingerprints and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary of computed metrics for the current batch.
        """
        fps, labels = batch  # Unpack input data
        out = self.forward(fps)  # Forward pass

        # Initialize lists for metrics calculation
        y = []  # True labels
        y_hat = []  # Predicted labels
        y_hat_proba = []  # Predicted probabilities

        for j, out in enumerate(out):
            mask = labels[:, j] >= 0.0  # Mask to exclude invalid labels
            y_pred = torch.where(out[mask] > 0.5, torch.ones(1), torch.zeros(1)).view(1, -1)  # Binarize predictions
            if y_pred.shape[1] > 0:  # Check if there are valid predictions
                y.extend(labels[:, j][mask].long().tolist())  # Collect true labels
                y_hat.extend(int(p[0]) for p in y_pred.view(-1, 1).tolist())  # Collect binary predictions
                y_hat_proba.extend(float(p[0]) for p in out[mask].view(-1, 1).tolist())  # Collect probabilities

        # Compute performance metrics
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        sens = tp / (tp + fn)  # Sensitivity
        spec = tn / (tn + fp)  # Specificity
        prec = tp / (tp + fp)  # Precision
        f1 = f1_score(y, y_hat)  # F1 Score
        acc = accuracy_score(y, y_hat)  # Accuracy
        mcc = matthews_corrcoef(y, y_hat)  # Matthews Correlation Coefficient
        auc = roc_auc_score(y, y_hat_proba)  # Area Under the ROC Curve

        # Package metrics into a dictionary
        metrics = {
            "test_acc": torch.tensor(acc),
            "test_sens": torch.tensor(sens),
            "test_spec": torch.tensor(spec),
            "test_prec": torch.tensor(prec),
            "test_f1": torch.tensor(f1),
            "test_mcc": torch.tensor(mcc),
            "test_auc": torch.tensor(auc),
        }

        # Log metrics for monitoring
        self.log_dict(metrics)
        self.test_step_outputs.append(metrics)  # Save metrics for post-processing
        return metrics

    def on_test_epoch_end(self):
        """
        Aggregate metrics across all test steps at the end of an epoch.

        Returns:
            dict: Aggregated metrics.
        """
        sums = Counter()  # Sum metrics across batches
        counters = Counter()  # Count occurrences of each metric
        for itemset in self.test_step_outputs:
            sums.update(itemset)  # Add metrics for the current step
            counters.update(itemset.keys())  # Track metric keys
        metrics = {x: float(sums[x]) / counters[x] for x in sums.keys()}  # Compute averages
        return metrics

if __name__ == "__main__":
    args = parse_args()

    chembl_version = args.chembl_version
    data_file = args.data_file
    batch_size = args.batch_size
    n_workers = args.n_workers
    lr = args.lr

    # Load weights and fingerprint length from the dataset file
    with tb.open_file(f"{data_file}", mode="r") as t_file:
        # Assign weights to tasks inversely proportional to their sample size.
        # Reference: https://ml.jku.at/publications/2014/NIPS2014f.pdf
        weights = t_file.root.weights[:]
        fps = t_file.root.fps
        fp_size = fps.shape[1]

    # Initialize dataset and create indices for splitting
    dataset = ChEMBLDataset(f"{data_file}")
    indices = list(range(len(dataset)))

    if args.cv_folds:
        # Dictionary to store metrics for each fold
        all_metrics = {
            "test_acc": [], "test_sens": [], "test_spec": [], "test_prec": [], 
            "test_f1": [], "test_mcc": [], "test_auc": []
        }

        # Perform k-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True)  # 5-fold CV with shuffling
        for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
            # Create data samplers for training and testing
            train_sampler = D.sampler.SubsetRandomSampler(train_idx)
            test_sampler = D.sampler.SubsetRandomSampler(test_idx)

            # Create data loaders for training and testing
            train_loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=n_workers, sampler=train_sampler
            )
            test_loader = DataLoader(
                dataset, batch_size=1000, num_workers=n_workers, sampler=test_sampler
            )

            # Initialize the multi-task model
            model = ChEMBLMultiTask(len(weights), fp_size, lr, weights)

            # Train the model using PyTorch Lightning
            trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu")
            trainer.fit(model, train_dataloaders=train_loader)

            # Evaluate the model on the test set
            mm = trainer.test(dataloaders=test_loader)

            # Collect metrics for the current fold
            for metric, value in mm[0].items():
                all_metrics[metric].append(value)

        # Save metrics from all folds to a JSON file
        with open(os.path.join(args.output_dir, f"chembl_{chembl_version}_metrics.json"), "w") as f:
            json.dump(all_metrics, f)

    # Train the model using the full dataset
    final_train_sampler = D.sampler.SubsetRandomSampler(indices)
    final_train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=final_train_sampler,
    )
    model = ChEMBLMultiTask(len(weights), fp_size, lr, weights)

    # Train the model with all available data
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu")
    trainer.fit(model, train_dataloaders=final_train_loader)

    # Extract target names from the dataset
    with tb.open_file(f"{data_file}", mode="r") as t_file:
        output_names = t_file.root.target_chembl_ids[:]

    # Save the trained model in PyTorch format
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"chembl_{chembl_version}_multitask.pth"))

    # Save the trained model in ONNX format
    model.to_onnx(
        os.path.join(args.output_dir, f"chembl_{chembl_version}_multitask.onnx"),
        torch.ones(fp_size),  # Example input for the model
        export_params=True,  # Include model parameters
        input_names=["input"],  # Name of input tensor
        output_names=output_names,  # Names of output tensors
    )

    # Quantize the ONNX model for optimized inference
    model_fp32 = os.path.join(args.output_dir, f"chembl_{chembl_version}_multitask.onnx")  # Path to FP32 model
    model_quant = os.path.join(args.output_dir, f"chembl_{chembl_version}_multitask_q8.onnx")  # Path to quantized model
    quantized_model = quantize_dynamic(model_fp32, model_quant)  # Perform quantization
