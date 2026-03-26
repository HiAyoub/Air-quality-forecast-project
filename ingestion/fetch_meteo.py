# ingestion/fetch_meteo.py
import logging
import os
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db.init import get_session, insert_measurement, upsert_station
from sqlalchemy import text

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Coordonnées centrales de chaque ville
VILLES_COORDS = {
    "Paris":       (48.8566,  2.3522),
    "Lyon":        (45.7640,  4.8357),
    "Marseille":   (43.2965,  5.3698),
    "Bordeaux":    (44.8378, -0.5792),
    "Lille":       (50.6292,  3.0573),
    "Toulouse":    (43.6047,  1.4442),
    "Strasbourg":  (48.5734,  7.7521),
}

METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "surface_pressure",
    "precipitation",
]


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_with_retry(url: str, params: dict, max_retries: int = 3) -> dict | None:
    """
    Appel GET avec retry automatique.

    Args:
        url: URL de l'endpoint.
        params: Paramètres de la requête.
        max_retries: Nombre maximum de tentatives.

    Returns:
        JSON de la réponse ou None en cas d'échec.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 503):
                wait = 2 ** attempt
                logger.warning(f"Rate limit ({response.status_code}), retry dans {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"Erreur HTTP {response.status_code} sur {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout, tentative {attempt + 1}/{max_retries}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Erreur connexion : {e}")
            return None
    return None


# ─── FETCH MÉTÉO ─────────────────────────────────────────────────────────────

def fetch_meteo_for_city(city: str, latitude: float, longitude: float) -> list[dict]:
    """
    Récupère les données météo horaires des dernières 24h pour une ville.

    Args:
        city: Nom de la ville.
        latitude: Latitude du centre-ville.
        longitude: Longitude du centre-ville.

    Returns:
        Liste de dicts {timestamp, temperature, humidity, wind_speed, pressure, precipitation}.
    """
    data = get_with_retry(
        url=OPEN_METEO_URL,
        params={
            "latitude":        latitude,
            "longitude":       longitude,
            "hourly":          ",".join(METEO_VARIABLES),
            "forecast_days":   2,
            "timezone":        "UTC",
            "timeformat":      "iso8601",
        },
    )

    if not data or "hourly" not in data:
        logger.warning(f"Pas de données météo pour {city}")
        return []

    hourly    = data["hourly"]
    timestamps = hourly.get("time", [])
    temps      = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    winds      = hourly.get("windspeed_10m", [])
    pressures  = hourly.get("surface_pressure", [])
    precips    = hourly.get("precipitation", [])

    records = []
    for i, ts_str in enumerate(timestamps):
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        records.append({
            "timestamp":   ts,
            "temperature": temps[i]      if i < len(temps)      else None,
            "humidity":    humidities[i] if i < len(humidities) else None,
            "wind_speed":  winds[i]      if i < len(winds)      else None,
            "pressure":    pressures[i]  if i < len(pressures)  else None,
            "precipitation": precips[i]  if i < len(precips)    else None,
        })

    logger.info(f"{city} : {len(records)} enregistrements météo récupérés")
    return records


# ─── MISE À JOUR EN BASE ──────────────────────────────────────────────────────

def update_meteo_in_db(city: str, records: list[dict]) -> int:
    """
    Met à jour les colonnes météo dans measurements_raw pour une ville.
    Joint sur timestamp_utc (heure exacte).

    Args:
        city: Nom de la ville.
        records: Liste de dicts météo avec timestamp.

    Returns:
        Nombre de lignes mises à jour.
    """
    sql = text("""
        UPDATE measurements_raw m
        SET
            temperature = :temperature,
            humidity    = :humidity,
            wind_speed  = :wind_speed,
            pressure    = :pressure
        FROM stations s
        WHERE m.station_id    = s.id
          AND s.city          = :city
          AND m.timestamp_utc = :timestamp
    """)

    updated = 0
    with get_session() as session:
        for record in records:
            result = session.execute(sql, {
                "city":        city,
                "timestamp":   record["timestamp"],
                "temperature": record["temperature"],
                "humidity":    record["humidity"],
                "wind_speed":  record["wind_speed"],
                "pressure":    record["pressure"],
            })
            updated += result.rowcount

    return updated


def insert_meteo_station(city: str, latitude: float, longitude: float, records: list[dict]) -> int:
    """
    Insère les données météo comme mesures standalone si aucune mesure
    pollution n'existe pour ce timestamp (utile pour avoir une base météo complète).

    Args:
        city: Nom de la ville.
        latitude: Latitude.
        longitude: Longitude.
        records: Liste de dicts météo.

    Returns:
        Nombre de lignes insérées.
    """
    station_id = upsert_station(
        name=f"Météo_{city}",
        city=city,
        latitude=latitude,
        longitude=longitude,
    )

    inserted = 0
    for record in records:
        ok = insert_measurement(
            station_id=station_id,
            timestamp_utc=record["timestamp"],
            temperature=record["temperature"],
            humidity=record["humidity"],
            wind_speed=record["wind_speed"],
            pressure=record["pressure"],
        )
        if ok:
            inserted += 1

    return inserted


# ─── PIPELINE PRINCIPAL ──────────────────────────────────────────────────────

def run_meteo_ingestion() -> None:
    """
    Pipeline complet météo :
    1. Récupère les données Open-Meteo pour chaque ville
    2. Met à jour les colonnes météo dans les mesures pollution existantes
    3. Insère les données météo standalone pour les timestamps sans pollution
    """
    logger.info("=== Démarrage ingestion Open-Meteo ===")
    total_updated  = 0
    total_inserted = 0

    for city, (lat, lon) in VILLES_COORDS.items():
        logger.info(f"--- {city} ---")

        records = fetch_meteo_for_city(city, lat, lon)
        if not records:
            continue

        # Étape 1 : mettre à jour les mesures pollution existantes
        updated = update_meteo_in_db(city, records)
        total_updated += updated
        logger.info(f"  {city} : {updated} mesure(s) pollution enrichie(s) avec météo")

        # Étape 2 : insérer les données météo standalone
        inserted = insert_meteo_station(city, lat, lon, records)
        total_inserted += inserted
        logger.info(f"  {city} : {inserted} enregistrement(s) météo standalone insérés")

        time.sleep(0.3)

    logger.info(f"=== Terminé : {total_updated} enrichies, {total_inserted} insérées ===")


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_meteo_ingestion()