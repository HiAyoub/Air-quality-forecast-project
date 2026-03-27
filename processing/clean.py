# processing/clean.py
import logging
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db.__init__ import get_session

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Ranges physiquement plausibles (µg/m³)
PM25_MIN,  PM25_MAX  = 0.0,  500.0
NO2_MIN,   NO2_MAX   = 0.0,  400.0
TEMP_MIN,  TEMP_MAX  = -30.0, 50.0
WIND_MIN,  WIND_MAX  = 0.0,  150.0
HUM_MIN,   HUM_MAX   = 0.0,  100.0

# Gap max pour interpolation (heures)
MAX_INTERPOLATION_GAP = 3


# ─── CHARGEMENT ──────────────────────────────────────────────────────────────

def load_measurements(city: str | None = None) -> pd.DataFrame:
    """
    Charge les mesures brutes depuis la base de données.

    Args:
        city: Filtrer par ville (None = toutes les villes).

    Returns:
        DataFrame avec toutes les mesures.
    """
    sql = """
        SELECT
            m.id, s.city, s.name as station_name,
            m.timestamp_utc, m.pm25, m.no2,
            m.temperature, m.humidity, m.wind_speed, m.pressure,
            m.is_outlier, m.is_imputed
        FROM measurements_raw m
        JOIN stations s ON s.id = m.station_id
        {where}
        ORDER BY s.city, m.timestamp_utc
    """
    where = f"WHERE s.city = '{city}'" if city else ""
    with get_session() as session:
        result = session.execute(text(sql.format(where=where)))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    logger.info(f"Chargé {len(df)} lignes depuis la base")
    return df


# ─── DÉTECTION OUTLIERS ──────────────────────────────────────────────────────

def flag_physical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flagge les valeurs hors des ranges physiquement plausibles.

    Args:
        df: DataFrame des mesures.

    Returns:
        DataFrame avec colonne is_outlier mise à jour.
    """
    mask = (
        (df["pm25"].notna()  & ((df["pm25"]  < PM25_MIN)  | (df["pm25"]  > PM25_MAX)))  |
        (df["no2"].notna()   & ((df["no2"]   < NO2_MIN)   | (df["no2"]   > NO2_MAX)))   |
        (df["temperature"].notna() & ((df["temperature"] < TEMP_MIN) | (df["temperature"] > TEMP_MAX))) |
        (df["wind_speed"].notna()  & ((df["wind_speed"]  < WIND_MIN) | (df["wind_speed"]  > WIND_MAX))) |
        (df["humidity"].notna()    & ((df["humidity"]    < HUM_MIN)  | (df["humidity"]    > HUM_MAX)))
    )
    n = mask.sum()
    df.loc[mask, "is_outlier"] = True
    logger.info(f"Outliers physiques flaggés : {n}")
    return df


def flag_iqr_outliers(df: pd.DataFrame, cols: list[str] = ["pm25", "no2"]) -> pd.DataFrame:
    """
    Flagge les outliers statistiques via la méthode IQR (1.5x) par ville.

    Args:
        df: DataFrame des mesures.
        cols: Colonnes à analyser.

    Returns:
        DataFrame avec is_outlier mis à jour.
    """
    total = 0
    for city in df["city"].unique():
        mask_city = df["city"] == city
        for col in cols:
            serie = df.loc[mask_city, col].dropna()
            if len(serie) < 4:
                continue
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr    = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask_outlier = mask_city & df[col].notna() & ((df[col] < lower) | (df[col] > upper))
            n = mask_outlier.sum()
            df.loc[mask_outlier, "is_outlier"] = True
            total += n

    logger.info(f"Outliers IQR flaggés : {total}")
    return df


# ─── IMPUTATION ──────────────────────────────────────────────────────────────

def interpolate_small_gaps(df: pd.DataFrame, cols: list[str] = ["pm25", "no2", "temperature", "wind_speed", "humidity"]) -> pd.DataFrame:
    """
    Interpole linéairement les gaps de <= MAX_INTERPOLATION_GAP heures
    consécutives. Au-delà, laisse les valeurs nulles.

    Args:
        df: DataFrame trié par city + timestamp_utc.
        cols: Colonnes à interpoler.

    Returns:
        DataFrame avec is_imputed mis à jour.
    """
    total_imputed = 0

    for city in df["city"].unique():
        mask_city = df["city"] == city
        df_city   = df.loc[mask_city].copy().sort_values("timestamp_utc")

        for col in cols:
            original_nulls = df_city[col].isna()
            if not original_nulls.any():
                continue

            # Interpolation uniquement sur les petits gaps
            interpolated = df_city[col].interpolate(
                method="linear",
                limit=MAX_INTERPOLATION_GAP,
                limit_direction="forward",
            )

            # Identifier ce qui a été imputé
            newly_filled = original_nulls & interpolated.notna()
            n = newly_filled.sum()

            if n > 0:
                df.loc[df_city.index[newly_filled], col]          = interpolated[newly_filled].values
                df.loc[df_city.index[newly_filled], "is_imputed"] = True
                total_imputed += n

    logger.info(f"Valeurs imputées par interpolation : {total_imputed}")
    return df


# ─── SAUVEGARDE EN BASE ───────────────────────────────────────────────────────

def save_flags_to_db(df: pd.DataFrame) -> None:
    """
    Sauvegarde les flags is_outlier et is_imputed en base de données.

    Args:
        df: DataFrame avec les flags mis à jour.
    """
    sql = text("""
        UPDATE measurements_raw
        SET is_outlier = :is_outlier,
            is_imputed = :is_imputed
        WHERE id = :id
    """)

    updated = 0
    with get_session() as session:
        for _, row in df.iterrows():
            session.execute(sql, {
                "id":         int(row["id"]),
                "is_outlier": bool(row["is_outlier"]),
                "is_imputed": bool(row["is_imputed"]),
            })
            updated += 1

    logger.info(f"Flags sauvegardés en base : {updated} lignes")


# ─── PIPELINE NETTOYAGE ──────────────────────────────────────────────────────

def run_cleaning(city: str | None = None, save: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage.

    Args:
        city: Ville à traiter (None = toutes).
        save: Sauvegarder les flags en base.

    Returns:
        DataFrame nettoyé.
    """
    logger.info("=== Démarrage nettoyage ===")

    df = load_measurements(city)

    if df.empty:
        logger.warning("Aucune donnée à nettoyer")
        return df

    df = flag_physical_outliers(df)
    df = flag_iqr_outliers(df)
    df = interpolate_small_gaps(df)

    n_outliers = df["is_outlier"].sum()
    n_imputed  = df["is_imputed"].sum()
    n_total    = len(df)

    logger.info(f"Résumé : {n_total} lignes | {n_outliers} outliers ({100*n_outliers/n_total:.1f}%) | {n_imputed} imputées ({100*n_imputed/n_total:.1f}%)")

    if save:
        save_flags_to_db(df)

    logger.info("=== Nettoyage terminé ===")
    return df


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_cleaning(save=True)
    print("\nAperçu des données nettoyées :")
    print(df[["city", "timestamp_utc", "pm25", "no2", "is_outlier", "is_imputed"]].head(10).to_string(index=False))