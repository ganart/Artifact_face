import torch
import torch.nn as nn
from torchmetrics.classification import F1Score
from torchvision.transforms import v2
from tqdm import tqdm
import pandas as pd
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer, device, model_type='cnn', model_name='Model'):
        """
            Args:
                    model (nn.Module): model for training (ResNet_Artifact, ViT_Artifact, EfficinetNet).
                    criterion (nn.Module): loss function.
                    optimizer (torch.optim.Optimizer): .
                    device (torch.device): select device (CPU или GPU).
                    model_type (str): type of model ('cnn' for cnn models, 'vit' for transformers) this need for great normalization.
                    model_name (str): name of model, using for log information
            """
        self.model = model
        self.model_type = model_type
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []
        self.f1_metric = F1Score(task='multiclass', num_classes=2, average='micro').to(device)
        self.model_name = model_name

    def train_epoch(self, dataloader):
        """
        Performs one epoch of learning.
        :param dataloader: DataLoader for training
        :return:  loss and F1 metrics per epoch
        """
        self.model.train()
        running_loss = 0.0
        self.f1_metric.reset()
        for images, labels, _ in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            images = self._normalize(images)
            torch.cuda.synchronize()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            self.f1_metric.update(preds, labels)
        epoch_loss = running_loss / len(dataloader)
        epoch_f1 = self.f1_metric.compute().item()
        return epoch_loss, epoch_f1

    def _normalize(self, images):
        """
                Apply different normalize function for different models type

                Args:
                    images (torch.Tensor):input image tensor (B, C, H, W).

                Returns:
                    torch.Tensor: normalized image.
        """
        if self.model_type == 'cnn':
            return v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(images)
        elif self.model_type == 'vit':
            return v2.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )(images)
        return images

    def validate_epoch(self, dataloader):
        """
        Performs one epoch of validation.
        :param dataloader: DataLoader for validation
        :return:  loss and F1 metrics per epoch
        """
        self.model.eval()
        running_loss = 0.0
        self.f1_metric.reset()
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                images = self._normalize(images)
                torch.cuda.synchronize()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                self.f1_metric.update(preds, labels)
        epoch_loss = running_loss / len(dataloader)
        epoch_f1 = self.f1_metric.compute().item()
        return epoch_loss, epoch_f1

    def test_epoch(self, test_loader, return_misclassified=False, print_misclassified=False):
        """

        :param test_loader: DataLoader test data
        :param return_misclassified: Return a list of classification errors.
        :param print_misclassified: Print paths to misclassified images.
        :return: Average loss and F1 metric. If return_misclassified=True, also returns a list of errors.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        misclassified = []
        self.f1_metric.reset()

        with torch.no_grad():
            for images, labels, paths in test_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                images = self._normalize(images)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                self.f1_metric.update(preds, labels)


                if return_misclassified or print_misclassified:
                    for pred, label, path in zip(preds.cpu().numpy(), labels.cpu().numpy(), paths):
                        if pred != label:
                            if print_misclassified:
                                print(
                                    f"Path: {path}, True mark: {label} ( 0=artifact, 1=no artifact), Predict: {pred}")
                            if return_misclassified:
                                misclassified.append({
                                    'path': path,
                                    'true_label': label,
                                    'pred_label': pred
                                })

        test_loss = total_loss / len(test_loader.dataset)
        test_f1 = self.f1_metric.compute().item()

        if return_misclassified:
            return test_loss, test_f1, misclassified
        return test_loss, test_f1

    def get_probabilities(self, dataloader):
        """
        Gets the class 1 (noartifact) probabilities for the ensemble
        :param dataloader: Load data
        :return: Class 1 probabilities for all images
        """
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for images, _, _ in tqdm(dataloader, desc=f"Getting Probabilities {self.model_name}"):
                images = images.to(self.device, non_blocking=True)
                images = self._normalize(images)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # probabilities of 1 class
                probabilities.extend(probs)
        return np.array(probabilities)

    def fit(self, train_loader, val_loader, epochs, stop_criteria=None):
        """
        Trains the model for a specified number of epochs, saving the best model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs.
            stop_criteria (float, optional): F1 score threshold for early stopping.

        Saves:
            - The best model is saved as 'best_{model_name}_model.pth'.
            - Metrics are saved in '{model_name}_metrics.csv'.
        """

        for epoch in range(epochs):
            train_loss, train_f1 = self.train_epoch(train_loader)
            val_loss, val_f1 = self.validate_epoch(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)
            print(f"[{self.model_name}] Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

            if len(self.val_f1s) == 1 or val_f1 > max(self.val_f1s[:-1]):
                torch.save(self.model.state_dict(), f'/kaggle/working/best_{self.model_name}_model.pth')
                print(f"[{self.model_name}] Best model was saved Val F1: {val_f1:.4f}")
            if stop_criteria is not None and val_f1 >= stop_criteria:
                print(
                    f"[{self.model_name}] Early stopping at epoch {epoch + 1} as Val F1 {val_f1:.4f} reached threshold {stop_criteria:.4f}")
                break

        # save metrics as CSV
        metrics_df = pd.DataFrame({
            'Epoch': range(1, epochs + 1),
            'Train Loss': self.train_losses,
            'Train F1': self.train_f1s,
            'Val Loss': self.val_losses,
            'Val F1': self.val_f1s
        })
        metrics_df.to_csv(f'/kaggle/working/{self.model_name}_metrics.csv', index=False)