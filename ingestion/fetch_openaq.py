# ingestion/fetch_openaq.py
import logging
import os
import sys
import time
from datetime import datetime

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

VILLES_FRANCE = {
    "Paris":       (48.815, 2.224,  48.902, 2.470),
    "Lyon":        (45.707, 4.771,  45.808, 4.898),
    "Marseille":   (43.169, 5.334,  43.381, 5.536),
    "Bordeaux":    (44.786, -0.638, 44.914, -0.527),
    "Lille":       (50.580, 2.972,  50.700, 3.130),
    "Toulouse":    (43.533, 1.350,  43.668, 1.506),
    "Strasbourg":  (48.530, 7.680,  48.620, 7.820),
}

HEADERS = {
    "Accept":    "application/json",
    "X-API-Key": OPENAQ_API_KEY,
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_with_retry(url: str, params: dict, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=15)
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


# ─── FETCH STATIONS ──────────────────────────────────────────────────────────

def fetch_stations_for_city(city: str, bbox: tuple) -> list[dict]:
    lat_min, lon_min, lat_max, lon_max = bbox
    data = get_with_retry(
        url=f"{OPENAQ_BASE_URL}/locations",
        params={"bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}", "limit": 20, "page": 1},
    )
    if not data or "results" not in data:
        logger.warning(f"Aucune station trouvée pour {city}")
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
    logger.info(f"{city} : {len(stations)} station(s) trouvée(s)")
    return stations


# ─── FETCH MESURES ───────────────────────────────────────────────────────────

def fetch_measurements_for_station(openaq_location_id: int) -> list[dict]:
    data = get_with_retry(
        url=f"{OPENAQ_BASE_URL}/locations/{openaq_location_id}/sensors",
        params={},
    )
    if not data or "results" not in data:
        return []
    mesures = []
    for sensor in data["results"]:
        param = sensor.get("parameter", {}).get("name", "").lower()
        if param not in ("pm25", "no2"):
            continue
        latest = sensor.get("latest") or {}
        value  = latest.get("value")
        dt     = latest.get("datetime", {})
        ts_str = dt.get("utc") if isinstance(dt, dict) else dt
        if value is None or not ts_str:
            continue
        mesures.append({"parameter": param, "value": value, "timestamp": ts_str})
    return mesures


# ─── PIPELINE ────────────────────────────────────────────────────────────────

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


def run_ingestion() -> None:
    logger.info("=== Démarrage ingestion OpenAQ ===")
    total_inserted = 0
    total_skipped  = 0

    for city, bbox in VILLES_FRANCE.items():
        logger.info(f"--- {city} ---")
        stations = fetch_stations_for_city(city, bbox)
        if not stations:
            continue

        for station in stations:
            station_id = upsert_station(
                name=station["name"], city=station["city"],
                latitude=station["latitude"], longitude=station["longitude"],
            )
            mesures = fetch_measurements_for_station(station["openaq_id"])
            if not mesures:
                logger.info(f"  {station['name']} : aucune mesure PM2.5/NO2")
                continue
            grouped = group_by_timestamp(mesures)
            for ts_str, values in grouped.items():
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    continue
                inserted = insert_measurement(
                    station_id=station_id, timestamp_utc=ts,
                    pm25=values["pm25"], no2=values["no2"],
                )
                if inserted:
                    total_inserted += 1
                else:
                    total_skipped += 1
            logger.info(f"  {station['name']} : {len(grouped)} mesure(s)")
        time.sleep(0.5)

    logger.info(f"=== Terminé : {total_inserted} insérées, {total_skipped} doublons ===")


if __name__ == "__main__":
    run_ingestion()