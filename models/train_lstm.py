# models/train_lstm.py
import logging
import os
import sys
from datetime import datetime

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing.features import build_features

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "air-quality-lstm"

SEQUENCE_LENGTH = 10       # nb de pas de temps en entrée
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 0.001
PATIENCE        = 10       # early stopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "pm25_lag_1d", "pm25_lag_2d",
    "no2_lag_1d",  "no2_lag_2d",
    "pm25_rolling_mean_3d",
    "no2_rolling_mean_3d",
    "temperature_norm", "humidity_norm",
    "wind_speed_norm",  "pressure_norm",
]

TARGETS = {
    "pm25_24h": "target_pm25_24h",
    "no2_24h":  "target_no2_24h",
}


# ─── DATASET ─────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Dataset PyTorch pour séries temporelles.
    Chaque sample est une séquence de SEQUENCE_LENGTH pas de temps.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


# ─── MODÈLE ──────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Architecture LSTM 2 couches avec dropout et couche linéaire finale.

    Args:
        input_size: Nombre de features en entrée.
        hidden_size: Taille de l'état caché.
        num_layers: Nombre de couches LSTM.
        dropout: Taux de dropout entre les couches.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out).squeeze(-1)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule RMSE, MAE et R²."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Prépare et normalise X et y.

    Args:
        df: DataFrame avec features et targets.
        target_col: Colonne cible.

    Returns:
        Tuple (X_scaled, y, scaler_X, scaler_y, feature_names).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset    = df[available + [target_col]].dropna()

    X = subset[available].values.astype(np.float32)
    y = subset[target_col].values.astype(np.float32)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y, available


# ─── ENTRAÎNEMENT ────────────────────────────────────────────────────────────

def train_lstm_for_target(
    df: pd.DataFrame,
    target_name: str,
    target_col: str,
) -> dict:
    """
    Entraîne un LSTM pour une cible avec early stopping et logging MLflow.

    Args:
        df: DataFrame avec features et targets.
        target_name: Nom court de la cible.
        target_col: Colonne target dans le DataFrame.

    Returns:
        Dict avec métriques et run_id MLflow.
    """
    logger.info(f"--- Entraînement LSTM : {target_name} ---")
    logger.info(f"Device : {DEVICE}")

    X, y, scaler_X, scaler_y, feature_names = prepare_data(df, target_col)

    if len(X) < SEQUENCE_LENGTH + 20:
        logger.warning(f"Pas assez de données ({len(X)} lignes)")
        return {}

    # Split temporel 80/20
    split     = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TimeSeriesDataset(X_train, y_train, SEQUENCE_LENGTH)
    val_ds   = TimeSeriesDataset(X_val,   y_val,   SEQUENCE_LENGTH)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMModel(len(feature_names), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"lstm_{target_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model_type": "lstm",
            "target":     target_name,
            "horizon":    target_name.split("_")[1],
            "pollutant":  target_name.split("_")[0],
        })
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size":     HIDDEN_SIZE,
            "num_layers":      NUM_LAYERS,
            "dropout":         DROPOUT,
            "batch_size":      BATCH_SIZE,
            "epochs":          EPOCHS,
            "learning_rate":   LEARNING_RATE,
            "patience":        PATIENCE,
            "n_features":      len(feature_names),
            "n_train":         len(X_train),
            "n_val":           len(X_val),
        })

        # Boucle d'entraînement
        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred      = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()

            train_loss /= max(len(train_dl), 1)
            val_loss   /= max(len(val_dl),   1)
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
            }, step=epoch)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    logger.info(f"  Early stopping à l'epoch {epoch+1}")
                    break

        # Évaluation finale avec le meilleur modèle
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                pred = model(X_batch.to(DEVICE)).cpu().numpy()
                all_preds.extend(pred)
                all_true.extend(y_batch.numpy())

        # Dénormaliser
        y_pred_real = scaler_y.inverse_transform(np.array(all_preds).reshape(-1, 1)).ravel()
        y_true_real = scaler_y.inverse_transform(np.array(all_true).reshape(-1, 1)).ravel()

        metrics = compute_metrics(y_true_real, y_pred_real)
        mlflow.log_metrics(metrics)

        logger.info(f"  RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f}")

        # Sauvegarder le modèle
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=f"lstm_{target_name}",
        )

        return {
            "target":  target_name,
            "run_id":  run.info.run_id,
            "metrics": metrics,
            "n_train": len(X_train),
        }


# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run_training() -> list[dict]:
    """Entraîne un LSTM pour PM2.5 J+1 et NO2 J+1."""
    logger.info("=== Démarrage entraînement LSTM ===")

    df = build_features(run_clean=True)
    if df.empty:
        logger.error("DataFrame vide")
        return []

    logger.info(f"Dataset : {df.shape[0]} lignes · {df.shape[1]} colonnes")

    results = []
    for target_name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        result = train_lstm_for_target(df, target_name, target_col)
        if result:
            results.append(result)

    logger.info("\n=== Résumé LSTM ===")
    for r in results:
        m = r["metrics"]
        logger.info(f"  {r['target']:12s} | RMSE: {m['rmse']:.3f} | MAE: {m['mae']:.3f} | R²: {m['r2']:.3f}")

    logger.info("=== Entraînement LSTM terminé ===")
    return results


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training()