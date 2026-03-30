# models/train_xgb.py
import logging
import os
import sys
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing.features import build_features

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "air-quality-xgboost"

TARGETS = {
    "pm25_24h": "target_pm25_24h",
    "no2_24h":  "target_no2_24h",
    "pm25_72h": "target_pm25_72h",
    "no2_72h":  "target_no2_72h",
}
"""
FEATURE_COLS = [
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos", "is_weekend",
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_7d",
    "no2_lag_1d",  "no2_lag_2d",  "no2_lag_7d",
    "pm25_rolling_mean_3d", "pm25_rolling_std_3d",
    "pm25_rolling_mean_7d", "pm25_rolling_std_7d",
    "no2_rolling_mean_3d",  "no2_rolling_std_3d",
    "no2_rolling_mean_7d",  "no2_rolling_std_7d",
    "temperature_norm", "humidity_norm", "wind_speed_norm", "pressure_norm",
    "wind_x_pm25", "humidity_x_pm25", "temp_x_no2",
]
"""
FEATURE_COLS = [
    # Temporelles — toujours disponibles
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "is_weekend",
    # Lags simples
    "pm25_lag_1d", "pm25_lag_2d",
    "no2_lag_1d",  "no2_lag_2d",
    # Rolling
    "pm25_rolling_mean_3d",
    "no2_rolling_mean_3d",
    # Météo — bien renseignée
    "temperature_norm", "humidity_norm",
    "wind_speed_norm",  "pressure_norm",
]

"""
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "random_state":     42,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,
}
"""
XGB_PARAMS = {
    "n_estimators":  200,
    "max_depth":     4,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "random_state":  42,
    "n_jobs":        -1,
}
N_SPLITS = 5


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
    Prépare X et y pour l'entraînement.
    Supprime les lignes avec des NaN dans les features ou la target.

    Args:
        df: DataFrame avec features et targets.
        target_col: Nom de la colonne cible.

    Returns:
        Tuple (X, y, feature_names).
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[available_features + [target_col]].dropna()

    X = subset[available_features].values
    y = subset[target_col].values

    logger.info(f"Données préparées : {len(X)} lignes · {len(available_features)} features")
    return X, y, available_features


# ─── ENTRAÎNEMENT ────────────────────────────────────────────────────────────

def train_xgboost_for_target(
    df: pd.DataFrame,
    target_name: str,
    target_col: str,
) -> dict:
    """
    Entraîne un modèle XGBoost pour une cible donnée avec
    validation croisée temporelle et logging MLflow.

    Args:
        df: DataFrame avec features et targets.
        target_name: Nom court de la cible (ex: pm25_24h).
        target_col: Nom de la colonne target dans le DataFrame.

    Returns:
        Dict avec métriques et run_id MLflow.
    """
    logger.info(f"--- Entraînement XGBoost : {target_name} ---")

    X, y, feature_names = prepare_data(df, target_col)
    if len(X) < 100:
        logger.warning(f"Pas assez de données ({len(X)} lignes) pour {target_name}")
        return {}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"xgb_{target_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tags({
            "model_type": "xgboost",
            "target":     target_name,
            "horizon":    target_name.split("_")[1],
            "pollutant":  target_name.split("_")[0],
        })

        # Params
        mlflow.log_params({**XGB_PARAMS, "n_splits": N_SPLITS, "n_features": len(feature_names)})

        # Cross-validation temporelle
        tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
        cv_rmse = []
        cv_mae  = []
        cv_r2   = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred   = model.predict(X_val)
            metrics  = compute_metrics(y_val, y_pred)
            cv_rmse.append(metrics["rmse"])
            cv_mae.append(metrics["mae"])
            cv_r2.append(metrics["r2"])

            logger.info(f"  Fold {fold+1}/{N_SPLITS} — RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f}")

        # Métriques moyennes CV
        mean_metrics = {
            "cv_rmse_mean": float(np.mean(cv_rmse)),
            "cv_rmse_std":  float(np.std(cv_rmse)),
            "cv_mae_mean":  float(np.mean(cv_mae)),
            "cv_r2_mean":   float(np.mean(cv_r2)),
        }
        mlflow.log_metrics(mean_metrics)

        # Modèle final sur toutes les données
        logger.info("  Entraînement modèle final sur toutes les données...")
        final_model = XGBRegressor(**{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"})
        final_model.fit(X, y, verbose=False)

        # Feature importance
        importance = dict(zip(feature_names, final_model.feature_importances_))
        top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("  Top 10 features :")
        for feat, imp in top10:
            logger.info(f"    {feat}: {imp:.4f}")

        # Log du modèle dans MLflow
        mlflow.xgboost.log_model(
            final_model,
            name="model",
            registered_model_name=f"xgb_{target_name}",
        )

        run_id = run.info.run_id
        logger.info(f"  Run MLflow : {run_id}")
        logger.info(f"  RMSE moyen CV : {mean_metrics['cv_rmse_mean']:.3f} ± {mean_metrics['cv_rmse_std']:.3f}")

        return {
            "target":    target_name,
            "run_id":    run_id,
            "metrics":   mean_metrics,
            "n_samples": len(X),
        }


# ─── PIPELINE COMPLET ────────────────────────────────────────────────────────

def run_training() -> list[dict]:
    """
    Entraîne un modèle XGBoost pour chaque cible (pm25/no2 × J+1/J+3).
    """
    logger.info("=== Démarrage entraînement XGBoost ===")

    logger.info("Chargement et feature engineering...")
    df = build_features(run_clean=True)

    if df.empty:
        logger.error("DataFrame vide après feature engineering")
        return []

    logger.info(f"Dataset : {df.shape[0]} lignes · {df.shape[1]} colonnes")

    results = []
    for target_name, target_col in TARGETS.items():
        if target_col not in df.columns:
            logger.warning(f"Colonne {target_col} absente, on passe")
            continue
        result = train_xgboost_for_target(df, target_name, target_col)
        if result:
            results.append(result)

    # Résumé final
    logger.info("\n=== Résumé des modèles entraînés ===")
    for r in results:
        m = r["metrics"]
        logger.info(f"  {r['target']:12s} | RMSE: {m['cv_rmse_mean']:.3f} ± {m['cv_rmse_std']:.3f} | MAE: {m['cv_mae_mean']:.3f} | R²: {m['cv_r2_mean']:.3f} | n={r['n_samples']}")

    logger.info("=== Entraînement terminé ===")
    return results


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_training()