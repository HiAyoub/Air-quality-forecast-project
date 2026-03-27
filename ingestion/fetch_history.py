# ingestion/fetch_history.py
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db.init import insert_measurement, upsert_station

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_API_KEY  = os.getenv("OPENAQ_API_KEY", "")

OPEN_METEO_URL  = "https://archive-api.open-meteo.com/v1/archive"

HEADERS = {
    "Accept":    "application/json",
    "X-API-Key": OPENAQ_API_KEY,
}

VILLES_FRANCE = {
    "Paris":       {"bbox": (48.815, 2.224,  48.902, 2.470),  "lat": 48.8566, "lon": 2.3522},
    "Lyon":        {"bbox": (45.707, 4.771,  45.808, 4.898),  "lat": 45.7640, "lon": 4.8357},
    "Marseille":   {"bbox": (43.169, 5.334,  43.381, 5.536),  "lat": 43.2965, "lon": 5.3698},
    "Bordeaux":    {"bbox": (44.786, -0.638, 44.914, -0.527), "lat": 44.8378, "lon": -0.5792},
    "Lille":       {"bbox": (50.580, 2.972,  50.700, 3.130),  "lat": 50.6292, "lon": 3.0573},
    "Toulouse":    {"bbox": (43.533, 1.350,  43.668, 1.506),  "lat": 43.6047, "lon": 1.4442},
    "Strasbourg":  {"bbox": (48.530, 7.680,  48.620, 7.820),  "lat": 48.5734, "lon": 7.7521},
}

# Période historique
DATE_START = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")
DATE_END   = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "surface_pressure",
    "precipitation",
]


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_with_retry(url: str, params: dict, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 503):
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limit ({response.status_code}), retry dans {wait}s...")
                time.sleep(wait)
                continue
            logger.error(f"Erreur HTTP {response.status_code} sur {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout, tentative {attempt + 1}/{max_retries}")
            time.sleep(2)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Erreur connexion : {e}")
            return None
    return None


# ─── FETCH STATIONS ──────────────────────────────────────────────────────────

def fetch_stations_for_city(city: str, bbox: tuple) -> list[dict]:
    lat_min, lon_min, lat_max, lon_max = bbox
    data = get_with_retry(
        url=f"{OPENAQ_BASE_URL}/locations",
        params={"bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "limit": 20},
    )
    if not data or "results" not in data:
        return []
    stations = []
    for loc in data["results"]:
        coords = loc.get("coordinates", {})
        if not coords.get("latitude") or not coords.get("longitude"):
            continue
        stations.append({
            "openaq_id": loc["id"],
            "name":      loc.get("name", f"Station_{loc['id']}"),
            "city":      city,
            "latitude":  coords["latitude"],
            "longitude": coords["longitude"],
        })
    logger.info(f"{city} : {len(stations)} station(s)")
    return stations


# ─── FETCH HISTORIQUE POLLUTION ───────────────────────────────────────────────

def fetch_historical_measurements(openaq_location_id: int, date_from: str, date_to: str) -> list[dict]:
    """
    Récupère l'historique des mesures PM2.5 et NO2 pour une station
    sur une période donnée, avec pagination automatique.

    Args:
        openaq_location_id: ID OpenAQ de la station.
        date_from: Date de début ISO8601.
        date_to: Date de fin ISO8601.

    Returns:
        Liste de mesures {parameter, value, timestamp}.
    """
    mesures  = []
    page     = 1
    limit    = 1000

    while True:
        data = get_with_retry(
            url=f"{OPENAQ_BASE_URL}/measurements",
            params={
                "locations_id": openaq_location_id,
                "date_from":    date_from,
                "date_to":      date_to,
                "limit":        limit,
                "page":         page,
                "parameters_id": "2,5",  # 2=pm25, 5=no2 dans OpenAQ
            },
        )

        if not data or "results" not in data or not data["results"]:
            break

        for m in data["results"]:
            param = m.get("parameter", "").lower()
            if param not in ("pm25", "no2"):
                continue
            ts_raw = m.get("date", {})
            ts_str = ts_raw.get("utc") if isinstance(ts_raw, dict) else None
            if not ts_str or m.get("value") is None:
                continue
            mesures.append({
                "parameter": param,
                "value":     m["value"],
                "timestamp": ts_str,
            })

        # Arrêter si moins de résultats que la limite (dernière page)
        if len(data["results"]) < limit:
            break

        page += 1
        time.sleep(0.2)

    return mesures


# ─── FETCH HISTORIQUE MÉTÉO ───────────────────────────────────────────────────

def fetch_historical_meteo(city: str, lat: float, lon: float, date_from: str, date_to: str) -> list[dict]:
    """
    Récupère l'historique météo horaire depuis Open-Meteo Archive API.

    Args:
        city: Nom de la ville.
        lat: Latitude.
        lon: Longitude.
        date_from: Date début (YYYY-MM-DD).
        date_to: Date fin (YYYY-MM-DD).

    Returns:
        Liste de dicts météo avec timestamp.
    """
    data = get_with_retry(
        url=OPEN_METEO_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": date_from[:10],
            "end_date":   date_to[:10],
            "hourly":     ",".join(METEO_VARIABLES),
            "timezone":   "UTC",
            "timeformat": "iso8601",
        },
    )

    if not data or "hourly" not in data:
        logger.warning(f"Pas de données météo historiques pour {city}")
        return []

    hourly     = data["hourly"]
    timestamps = hourly.get("time", [])
    temps      = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    winds      = hourly.get("windspeed_10m", [])
    pressures  = hourly.get("surface_pressure", [])

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
        })

    logger.info(f"{city} météo historique : {len(records)} enregistrements")
    return records


# ─── GROUPER PAR TIMESTAMP ────────────────────────────────────────────────────

def group_by_timestamp(mesures: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for m in mesures:
        ts = m["timestamp"]
        if not ts:
            continue
        if ts not in grouped:
            grouped[ts] = {"pm25": None, "no2": None}
        if m["parameter"] == "pm25":
            grouped[ts]["pm25"] = m["value"]
        elif m["parameter"] == "no2":
            grouped[ts]["no2"] = m["value"]
    return grouped


# ─── PIPELINE PRINCIPAL ──────────────────────────────────────────────────────

def run_historical_ingestion() -> None:
    """
    Récupère 90 jours d'historique pollution + météo pour toutes les villes.
    """
    logger.info(f"=== Ingestion historique du {DATE_START[:10]} au {DATE_END[:10]} ===")
    total_pollution = 0
    total_meteo     = 0

    for city, config in VILLES_FRANCE.items():
        logger.info(f"--- {city} ---")
        bbox = config["bbox"]
        lat  = config["lat"]
        lon  = config["lon"]

        # 1. Stations
        stations = fetch_stations_for_city(city, bbox)
        if not stations:
            continue

        # 2. Historique pollution par station
        for station in stations:
            station_id = upsert_station(
                name=station["name"], city=station["city"],
                latitude=station["latitude"], longitude=station["longitude"],
            )

            mesures = fetch_historical_measurements(
                openaq_location_id=station["openaq_id"],
                date_from=DATE_START,
                date_to=DATE_END,
            )

            if not mesures:
                logger.info(f"  {station['name']} : aucun historique")
                continue

            grouped = group_by_timestamp(mesures)
            inserted = 0
            for ts_str, values in grouped.items():
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    continue
                ok = insert_measurement(
                    station_id=station_id,
                    timestamp_utc=ts,
                    pm25=values["pm25"],
                    no2=values["no2"],
                )
                if ok:
                    inserted += 1

            total_pollution += inserted
            logger.info(f"  {station['name']} : {inserted} mesures historiques insérées")
            time.sleep(0.3)

        # 3. Historique météo
        meteo_records = fetch_historical_meteo(city, lat, lon, DATE_START, DATE_END)
        meteo_station_id = upsert_station(
            name=f"Météo_{city}", city=city, latitude=lat, longitude=lon,
        )
        meteo_inserted = 0
        for record in meteo_records:
            ok = insert_measurement(
                station_id=meteo_station_id,
                timestamp_utc=record["timestamp"],
                temperature=record["temperature"],
                humidity=record["humidity"],
                wind_speed=record["wind_speed"],
                pressure=record["pressure"],
            )
            if ok:
                meteo_inserted += 1
        total_meteo += meteo_inserted
        logger.info(f"  {city} météo : {meteo_inserted} enregistrements insérés")

        time.sleep(1)

    logger.info(f"=== Terminé : {total_pollution} mesures pollution + {total_meteo} météo ===")


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_historical_ingestion()