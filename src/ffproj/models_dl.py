"""
Deep Learning models with embeddings and quantile regression for fantasy projections.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .config import (
    DL_EMBEDDING_DIM, DL_HIDDEN_DIM, DL_TCN_KERNEL, DL_TCN_LEVELS,
    DL_SEQUENCE_LENGTH, DL_BATCH_SIZE, DL_LEARNING_RATE, DL_NUM_EPOCHS, DL_PATIENCE
)
from .utils import get_device, enforce_monotonic_quantiles


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Pinball loss (quantile loss) for quantile regression.

    Parameters
    ----------
    y_true : torch.Tensor
        True values
    y_pred : torch.Tensor
        Predicted values
    tau : float
        Quantile (e.g., 0.1, 0.5, 0.9)

    Returns
    -------
    torch.Tensor
        Pinball loss
    """
    error = y_true - y_pred
    return torch.mean(torch.maximum(tau * error, (tau - 1) * error))


class FantasyDataset(Dataset):
    """
    PyTorch dataset for fantasy football time series data.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        categorical_ids: Dict[str, np.ndarray] = None
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        sequences : np.ndarray
            Shape: (num_samples, sequence_length, num_features)
        targets : np.ndarray
            Shape: (num_samples,)
        categorical_ids : dict, optional
            Dictionary of categorical variable name -> ids array
            Each array shape: (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

        self.categorical_ids = {}
        if categorical_ids:
            for name, ids in categorical_ids.items():
                self.categorical_ids[name] = torch.LongTensor(ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            'sequence': self.sequences[idx],
            'target': self.targets[idx]
        }

        for name, ids in self.categorical_ids.items():
            item[name] = ids[idx]

        return item


class EmbeddingModule(nn.Module):
    """
    Embedding module for categorical variables.
    """

    def __init__(self, cardinalities: Dict[str, int], embedding_dim: int = 16):
        """
        Initialize embeddings.

        Parameters
        ----------
        cardinalities : dict
            Dictionary of variable name -> number of unique values
        embedding_dim : int
            Embedding dimension (will be adjusted per variable)
        """
        super().__init__()
        self.emb_layers = nn.ModuleDict()

        for name, num_categories in cardinalities.items():
            # Embedding dimension: min of specified dim or heuristic based on cardinality
            emb_dim = min(embedding_dim, max(4, int(num_categories ** 0.25) * 2))
            self.emb_layers[name] = nn.Embedding(num_categories, emb_dim)

    def forward(self, categorical_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: embed all categorical variables and concatenate.

        Parameters
        ----------
        categorical_ids : dict
            Dictionary of variable name -> id tensor

        Returns
        -------
        torch.Tensor
            Concatenated embeddings, shape: (batch_size, total_emb_dim)
        """
        embeddings = []
        for name, ids in categorical_ids.items():
            if name in self.emb_layers:
                embeddings.append(self.emb_layers[name](ids))

        if embeddings:
            return torch.cat(embeddings, dim=-1)
        else:
            return None


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        num_levels: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize TCN.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dim : int
            Hidden dimension
        kernel_size : int
            Convolution kernel size
        num_levels : int
            Number of TCN levels (dilation increases exponentially)
        dropout : float
            Dropout rate
        """
        super().__init__()

        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        layers = []
        dilation = 1
        for _ in range(num_levels):
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
            layers.extend([
                conv,
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            dilation *= 2

        self.tcn = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input shape: (batch_size, sequence_length, input_dim)

        Returns
        -------
        torch.Tensor
            Output shape: (batch_size, hidden_dim)
        """
        # Transpose to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_conv(x)

        # TCN layers
        x = self.tcn(x)

        # Take last time step
        x = x[:, :, -1]

        return x


class FantasyTCN(nn.Module):
    """
    TCN-based model for fantasy projections with quantile heads.
    """

    def __init__(
        self,
        sequence_feature_dim: int,
        cardinalities: Dict[str, int] = None,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        tcn_kernel: int = 3,
        tcn_levels: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize Fantasy TCN model.

        Parameters
        ----------
        sequence_feature_dim : int
            Number of features in the sequence
        cardinalities : dict, optional
            Cardinalities of categorical variables for embeddings
        embedding_dim : int
            Embedding dimension
        hidden_dim : int
            Hidden dimension
        tcn_kernel : int
            TCN kernel size
        tcn_levels : int
            Number of TCN levels
        dropout : float
            Dropout rate
        """
        super().__init__()

        # Embeddings
        self.use_embeddings = cardinalities is not None
        if self.use_embeddings:
            self.embeddings = EmbeddingModule(cardinalities, embedding_dim)
            # Calculate total embedding dimension
            total_emb_dim = sum(
                min(embedding_dim, max(4, int(card ** 0.25) * 2))
                for card in cardinalities.values()
            )
        else:
            total_emb_dim = 0

        # TCN
        self.tcn = TemporalConvNet(
            input_dim=sequence_feature_dim + total_emb_dim,
            hidden_dim=hidden_dim,
            kernel_size=tcn_kernel,
            num_levels=tcn_levels,
            dropout=dropout
        )

        # Quantile heads
        self.head_q10 = nn.Linear(hidden_dim, 1)
        self.head_q50 = nn.Linear(hidden_dim, 1)
        self.head_q90 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        categorical_ids: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence, shape: (batch_size, seq_len, num_features)
        categorical_ids : dict, optional
            Categorical variable ids

        Returns
        -------
        tuple
            (q10, q50, q90) predictions, each shape: (batch_size, 1)
        """
        batch_size, seq_len, _ = sequence.shape

        # Get embeddings if available
        if self.use_embeddings and categorical_ids:
            emb = self.embeddings(categorical_ids)  # (batch, emb_dim)
            # Expand to sequence length
            emb = emb.unsqueeze(1).expand(-1, seq_len, -1)
            # Concatenate with sequence
            sequence = torch.cat([sequence, emb], dim=-1)

        # TCN
        h = self.tcn(sequence)  # (batch, hidden_dim)

        # Quantile predictions
        q10 = self.head_q10(h)
        q50 = self.head_q50(h)
        q90 = self.head_q90(h)

        return q10, q50, q90


class FantasyLSTM(nn.Module):
    """
    LSTM-based model for fantasy projections with quantile heads.
    """

    def __init__(
        self,
        sequence_feature_dim: int,
        cardinalities: Dict[str, int] = None,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Fantasy LSTM model.

        Parameters
        ----------
        sequence_feature_dim : int
            Number of features in the sequence
        cardinalities : dict, optional
            Cardinalities of categorical variables
        embedding_dim : int
            Embedding dimension
        hidden_dim : int
            LSTM hidden dimension
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        """
        super().__init__()

        # Embeddings
        self.use_embeddings = cardinalities is not None
        if self.use_embeddings:
            self.embeddings = EmbeddingModule(cardinalities, embedding_dim)
            total_emb_dim = sum(
                min(embedding_dim, max(4, int(card ** 0.25) * 2))
                for card in cardinalities.values()
            )
        else:
            total_emb_dim = 0

        # LSTM
        self.lstm = nn.LSTM(
            input_size=sequence_feature_dim + total_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Quantile heads
        self.head_q10 = nn.Linear(hidden_dim, 1)
        self.head_q50 = nn.Linear(hidden_dim, 1)
        self.head_q90 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        categorical_ids: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence, shape: (batch, seq_len, features)
        categorical_ids : dict, optional
            Categorical variable ids

        Returns
        -------
        tuple
            (q10, q50, q90) predictions
        """
        batch_size, seq_len, _ = sequence.shape

        # Embeddings
        if self.use_embeddings and categorical_ids:
            emb = self.embeddings(categorical_ids)
            emb = emb.unsqueeze(1).expand(-1, seq_len, -1)
            sequence = torch.cat([sequence, emb], dim=-1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        # Use last hidden state
        h = h_n[-1]  # (batch, hidden_dim)

        # Quantile predictions
        q10 = self.head_q10(h)
        q50 = self.head_q50(h)
        q90 = self.head_q90(h)

        return q10, q50, q90


class FantasyModelTrainer:
    """
    Trainer for fantasy DL models with quantile loss.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = DL_LEARNING_RATE,
        quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            Fantasy model (TCN or LSTM)
        device : torch.device, optional
            Device to train on
        learning_rate : float
            Learning rate
        quantiles : tuple
            Target quantiles (default: 0.1, 0.5, 0.9)
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.quantiles = quantiles

        print(f"Training on device: {self.device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move to device
            sequence = batch['sequence'].to(self.device)
            target = batch['target'].to(self.device).unsqueeze(1)

            categorical_ids = {
                name: batch[name].to(self.device)
                for name in batch if name not in ['sequence', 'target']
            }

            # Forward pass
            q10, q50, q90 = self.model(sequence, categorical_ids if categorical_ids else None)

            # Compute quantile losses
            loss_q10 = pinball_loss(target, q10, self.quantiles[0])
            loss_q50 = pinball_loss(target, q50, self.quantiles[1])
            loss_q90 = pinball_loss(target, q90, self.quantiles[2])

            loss = loss_q10 + loss_q50 + loss_q90

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, np.ndarray]]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0.0

        all_targets = []
        all_q10 = []
        all_q50 = []
        all_q90 = []

        with torch.no_grad():
            for batch in dataloader:
                sequence = batch['sequence'].to(self.device)
                target = batch['target'].to(self.device).unsqueeze(1)

                categorical_ids = {
                    name: batch[name].to(self.device)
                    for name in batch if name not in ['sequence', 'target']
                }

                q10, q50, q90 = self.model(sequence, categorical_ids if categorical_ids else None)

                loss_q10 = pinball_loss(target, q10, self.quantiles[0])
                loss_q50 = pinball_loss(target, q50, self.quantiles[1])
                loss_q90 = pinball_loss(target, q90, self.quantiles[2])

                loss = loss_q10 + loss_q50 + loss_q90
                total_loss += loss.item()

                all_targets.append(target.cpu().numpy())
                all_q10.append(q10.cpu().numpy())
                all_q50.append(q50.cpu().numpy())
                all_q90.append(q90.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        predictions = {
            'target': np.concatenate(all_targets).flatten(),
            'q10': np.concatenate(all_q10).flatten(),
            'q50': np.concatenate(all_q50).flatten(),
            'q90': np.concatenate(all_q90).flatten()
        }

        return avg_loss, predictions

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = DL_NUM_EPOCHS,
        patience: int = DL_PATIENCE,
        verbose: bool = True
    ) -> Dict:
        """
        Train model with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Maximum number of epochs
        patience : int
            Early stopping patience
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _ = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        return history

    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Make predictions on new data."""
        _, predictions = self.evaluate(dataloader)
        return predictions
