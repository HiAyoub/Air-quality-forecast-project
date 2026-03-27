# ingestion/fetch_history.py
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from sqlalchemy import text

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db.__init__ import get_session, insert_measurement, upsert_station

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

DATE_START = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
DATE_END   = datetime.utcnow().strftime("%Y-%m-%d")

VILLES_COORDS = {
    "Paris":      {"lat": 48.8566, "lon": 2.3522},
    "Lyon":       {"lat": 45.7640, "lon": 4.8357},
    "Marseille":  {"lat": 43.2965, "lon": 5.3698},
    "Bordeaux":   {"lat": 44.8378, "lon": -0.5792},
    "Lille":      {"lat": 50.6292, "lon": 3.0573},
    "Toulouse":   {"lat": 43.6047, "lon": 1.4442},
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521},
}

METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "surface_pressure",
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
                logger.warning(f"Rate limit, retry dans {wait}s...")
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


# ─── LECTURE STATIONS EN BASE ────────────────────────────────────────────────

def get_stations_from_db() -> list[dict]:
    """
    Lit toutes les stations avec openaq_id depuis la base.
    On utilise ces IDs pour requêter l'historique OpenAQ.
    """
    sql = text("""
        SELECT id, name, city, openaq_id, latitude, longitude
        FROM stations
        WHERE openaq_id IS NOT NULL
        ORDER BY city, name
    """)
    with get_session() as session:
        result = session.execute(sql)
        rows = [dict(zip(result.keys(), row)) for row in result.fetchall()]
    logger.info(f"{len(rows)} stations avec openaq_id trouvées en base")
    return rows


# ─── FETCH HISTORIQUE POLLUTION ───────────────────────────────────────────────

def fetch_sensors_for_location(openaq_id: int) -> list[int]:
    """
    Récupère les sensor_ids d'une station pour pm25 et no2.
    Dans l'API v3, les mesures historiques passent par /sensors/{id}/measurements.
    """
    data = get_with_retry(
        url=f"{OPENAQ_BASE_URL}/locations/{openaq_id}/sensors",
        params={},
    )
    if not data or "results" not in data:
        return []

    sensor_ids = []
    for sensor in data["results"]:
        param = sensor.get("parameter", {}).get("name", "").lower()
        if param in ("pm25", "no2", "pm10"):  # Parfois PM10 contient PM2.5
            sensor_ids.append({
                "sensor_id": sensor["id"],
                "parameter": "pm25" if param == "pm10" else param,
            })
    return sensor_ids


def fetch_sensor_history(sensor_id: int, parameter: str, date_from: str, date_to: str) -> list[dict]:
    """
    Récupère l'historique d'un sensor par tranches de 7 jours
    pour éviter les timeouts sur les longues périodes.
    """
    mesures  = []
    start    = datetime.strptime(date_from, "%Y-%m-%d")
    end      = datetime.strptime(date_to,   "%Y-%m-%d")
    chunk    = timedelta(days=7)
    current  = start

    while current < end:
        chunk_end = min(current + chunk, end)

        data = get_with_retry(
            url=f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
            params={
                "date_from": current.strftime("%Y-%m-%dT00:00:00Z"),
                "date_to":   chunk_end.strftime("%Y-%m-%dT23:59:59Z"),
                "limit":     1000,
                "page":      1,
                "order_by":  "datetime",
                "order":     "asc",
            },
        )

        if data and "results" in data:
            for m in data["results"]:
                value  = m.get("value")
                dt     = m.get("period", {}).get("datetimeTo", {})
                ts_str = dt.get("utc") if isinstance(dt, dict) else None
                if value is None or not ts_str:
                    continue
                mesures.append({
                    "parameter": parameter,
                    "value":     value,
                    "timestamp": ts_str,
                })

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return mesures


# ─── FETCH HISTORIQUE MÉTÉO ───────────────────────────────────────────────────

def fetch_historical_meteo(city: str, lat: float, lon: float) -> list[dict]:
    """Récupère l'historique météo horaire depuis Open-Meteo Archive API."""
    data = get_with_retry(
        url=OPEN_METEO_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": DATE_START,
            "end_date":   DATE_END,
            "hourly":     ",".join(METEO_VARIABLES),
            "timezone":   "UTC",
        },
    )

    if not data or "hourly" not in data:
        logger.warning(f"Pas de données météo historiques pour {city}")
        return []

    hourly     = data["hourly"]
    timestamps = hourly.get("time", [])
    records    = []

    for i, ts_str in enumerate(timestamps):
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        records.append({
            "timestamp":   ts,
            "temperature": hourly.get("temperature_2m",      [None] * (i+1))[i],
            "humidity":    hourly.get("relative_humidity_2m",[None] * (i+1))[i],
            "wind_speed":  hourly.get("windspeed_10m",       [None] * (i+1))[i],
            "pressure":    hourly.get("surface_pressure",    [None] * (i+1))[i],
        })

    logger.info(f"{city} météo : {len(records)} enregistrements")
    return records


# ─── PIPELINE PRINCIPAL ──────────────────────────────────────────────────────

def run_historical_ingestion() -> None:
    """
    Pipeline historique complet :
    1. Lit les stations depuis la base (avec openaq_id)
    2. Pour chaque station : récupère les sensors puis leur historique
    3. Récupère l'historique météo par ville
    """
    logger.info(f"=== Ingestion historique {DATE_START} → {DATE_END} ===")
    total_pollution = 0
    total_meteo     = 0

    stations = get_stations_from_db()

    # ── Pollution ──
    for station in stations:
        # Ignorer les stations météo standalone
        if station["name"].startswith("Météo_"):
            continue

        logger.info(f"  {station['city']} | {station['name']} (openaq_id={station['openaq_id']})")

        sensors = fetch_sensors_for_location(station["openaq_id"])
        if not sensors:
            logger.info(f"    Aucun sensor PM2.5/NO2")
            continue

        # Regrouper les mesures par timestamp
        grouped: dict[str, dict] = {}
        for sensor in sensors:
            mesures = fetch_sensor_history(
                sensor_id=sensor["sensor_id"],
                parameter=sensor["parameter"],
                date_from=DATE_START,
                date_to=DATE_END,
            )
            for m in mesures:
                ts = m["timestamp"]
                if ts not in grouped:
                    grouped[ts] = {"pm25": None, "no2": None}
                grouped[ts][m["parameter"]] = m["value"]

        # Insérer en base
        inserted = 0
        for ts_str, values in grouped.items():
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
            ok = insert_measurement(
                station_id=station["id"],
                timestamp_utc=ts,
                pm25=values.get("pm25"),
                no2=values.get("no2"),
            )
            if ok:
                inserted += 1

        total_pollution += inserted
        logger.info(f"    {inserted} mesures historiques insérées")
        time.sleep(0.5)

    # ── Météo par ville ──
    logger.info("--- Ingestion météo historique ---")
    villes_traitees = set()

    for station in stations:
        city = station["city"]
        if city in villes_traitees or city not in VILLES_COORDS:
            continue
        villes_traitees.add(city)

        coords   = VILLES_COORDS[city]
        records  = fetch_historical_meteo(city, coords["lat"], coords["lon"])
        meteo_id = upsert_station(
            name=f"Météo_{city}", city=city,
            latitude=coords["lat"], longitude=coords["lon"],
        )
        inserted = 0
        for r in records:
            ok = insert_measurement(
                station_id=meteo_id,
                timestamp_utc=r["timestamp"],
                temperature=r["temperature"],
                humidity=r["humidity"],
                wind_speed=r["wind_speed"],
                pressure=r["pressure"],
            )
            if ok:
                inserted += 1
        total_meteo += inserted
        logger.info(f"  {city} météo : {inserted} enregistrements insérés")
        time.sleep(0.5)

    logger.info(f"=== Terminé : {total_pollution} pollution + {total_meteo} météo ===")


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_historical_ingestion()