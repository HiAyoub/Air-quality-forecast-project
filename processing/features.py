# processing/features.py
import logging
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from processing.clean import run_cleaning

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

LAG_HOURS        = [1, 3, 6, 24]
ROLLING_WINDOWS  = [6, 24]
TARGET_COLS      = ["pm25", "no2"]
METEO_COLS       = ["temperature", "humidity", "wind_speed", "pressure"]


# ─── FEATURES TEMPORELLES ────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features temporelles : heure, jour, mois, weekend,
    et encodage cyclique sin/cos pour capturer la périodicité.

    Args:
        df: DataFrame avec colonne timestamp_utc.

    Returns:
        DataFrame enrichi.
    """
    df = df.copy()
    ts = df["timestamp_utc"]

    df["hour"]       = ts.dt.hour
    df["dayofweek"]  = ts.dt.dayofweek
    df["month"]      = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Encodage cyclique — transforme heure 0 et heure 23 en voisins
    df["hour_sin"]      = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour"]      / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"]     / 12)

    logger.info("Features temporelles ajoutées : hour, dayofweek, month, is_weekend + sin/cos")
    return df


# ─── FEATURES DE LAG ─────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les valeurs passées (lags) pour PM2.5 et NO2.
    Calculé par groupe (city + station_name) pour éviter les fuites entre stations.

    Args:
        df: DataFrame trié par city + timestamp_utc.

    Returns:
        DataFrame avec colonnes lag_Xh pour pm25 et no2.
    """
    df = df.copy().sort_values(["city", "station_name", "timestamp_utc"])

    for col in TARGET_COLS:
        for lag in LAG_HOURS:
            feat_name = f"{col}_lag_{lag}h"
            df[feat_name] = (
                df.groupby(["city", "station_name"])[col]
                .shift(lag)
            )

    n_lags = len(TARGET_COLS) * len(LAG_HOURS)
    logger.info(f"Features de lag ajoutées : {n_lags} colonnes ({LAG_HOURS}h pour {TARGET_COLS})")
    return df


# ─── ROLLING STATISTICS ──────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les statistiques glissantes (moyenne et écart-type)
    sur fenêtres de 6h et 24h pour PM2.5 et NO2.

    Args:
        df: DataFrame trié par city + timestamp_utc.

    Returns:
        DataFrame avec colonnes rolling_mean_Xh et rolling_std_Xh.
    """
    df = df.copy().sort_values(["city", "station_name", "timestamp_utc"])

    for col in TARGET_COLS:
        for window in ROLLING_WINDOWS:
            grp = df.groupby(["city", "station_name"])[col]

            df[f"{col}_rolling_mean_{window}h"] = (
                grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"{col}_rolling_std_{window}h"] = (
                grp.transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )

    n_rolling = len(TARGET_COLS) * len(ROLLING_WINDOWS) * 2
    logger.info(f"Features rolling ajoutées : {n_rolling} colonnes (fenêtres {ROLLING_WINDOWS}h)")
    return df


# ─── FEATURES MÉTÉO CROISÉES ─────────────────────────────────────────────────

def add_meteo_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features d'interaction entre météo et pollution.
    - wind_x_pm25 : vent fort = meilleure dispersion des particules
    - humidity_x_pm25 : humidité élevée = particules hygroscopiques plus lourdes

    Args:
        df: DataFrame avec colonnes météo.

    Returns:
        DataFrame avec features croisées.
    """
    df = df.copy()

    if "wind_speed" in df.columns and "pm25" in df.columns:
        df["wind_x_pm25"] = df["wind_speed"] * df["pm25"].fillna(0)

    if "humidity" in df.columns and "pm25" in df.columns:
        df["humidity_x_pm25"] = df["humidity"] * df["pm25"].fillna(0)

    if "temperature" in df.columns and "no2" in df.columns:
        df["temp_x_no2"] = df["temperature"] * df["no2"].fillna(0)

    logger.info("Features météo croisées ajoutées : wind_x_pm25, humidity_x_pm25, temp_x_no2")
    return df


# ─── NORMALISATION MÉTÉO ─────────────────────────────────────────────────────

def normalize_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les features météo par standardisation (z-score) globale.
    Les paramètres sont calculés sur le jeu d'entraînement uniquement.

    Args:
        df: DataFrame avec colonnes météo.

    Returns:
        DataFrame avec colonnes météo normalisées (suffixe _norm).
    """
    df = df.copy()

    for col in METEO_COLS:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std  = df[col].std()
        if std > 0:
            df[f"{col}_norm"] = (df[col] - mean) / std
        else:
            df[f"{col}_norm"] = 0.0

    logger.info(f"Normalisation météo : {METEO_COLS}")
    return df


# ─── TARGETS ─────────────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame, horizons: list[int] = [24, 72]) -> pd.DataFrame:
    """
    Ajoute les variables cibles : valeur de PM2.5 et NO2
    à horizon H heures dans le futur (J+1 = 24h, J+3 = 72h).

    Args:
        df: DataFrame trié par city + timestamp_utc.
        horizons: Liste des horizons en heures.

    Returns:
        DataFrame avec colonnes target_pm25_Xh et target_no2_Xh.
    """
    df = df.copy().sort_values(["city", "station_name", "timestamp_utc"])

    for horizon in horizons:
        for col in TARGET_COLS:
            target_name = f"target_{col}_{horizon}h"
            df[target_name] = (
                df.groupby(["city", "station_name"])[col]
                .shift(-horizon)
            )

    logger.info(f"Targets ajoutées : horizons {horizons}h pour {TARGET_COLS}")
    return df


# ─── PIPELINE COMPLET ─────────────────────────────────────────────────────────

def build_features(city: str | None = None, run_clean: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering.

    Args:
        city: Ville à traiter (None = toutes).
        run_clean: Lancer le nettoyage avant le feature engineering.

    Returns:
        DataFrame prêt pour l'entraînement ML.
    """
    logger.info("=== Démarrage feature engineering ===")

    if run_clean:
        df = run_cleaning(city=city, save=False)
    else:
        from processing.clean import load_measurements
        df = load_measurements(city)

    if df.empty:
        logger.warning("DataFrame vide, feature engineering annulé")
        return df

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_meteo_interaction_features(df)
    df = normalize_meteo(df)
    df = add_targets(df, horizons=[1, 3, 24, 72])


    # Supprimer les lignes sans target (dernières heures — futur inconnu)
    # Par
    before = len(df)
    df = df.dropna(subset=["target_pm25_1h", "target_no2_1h"], how="all")
    after  = len(df)
    logger.info(f"Lignes supprimées sans target : {before - after}")

    n_features = len([c for c in df.columns if c not in
                      ["id", "city", "station_name", "timestamp_utc",
                       "is_outlier", "is_imputed"]])
    logger.info(f"=== Feature engineering terminé : {len(df)} lignes · {n_features} features ===")
    return df


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_features()

    print("\nColonnes générées :")
    feature_cols = [c for c in df.columns if c not in
                    ["id", "city", "station_name", "timestamp_utc", "is_outlier", "is_imputed"]]
    for col in sorted(feature_cols):
        print(f"  {col}")

    print(f"\nShape final : {df.shape}")
    print(f"\nAperçu (Paris) :")
    paris = df[df["city"] == "Paris"][["timestamp_utc", "pm25", "no2",
                                        "pm25_lag_1h", "pm25_rolling_mean_6h",
                                        "hour_sin", "target_pm25_24h"]].head(5)
    print(paris.to_string(index=False))