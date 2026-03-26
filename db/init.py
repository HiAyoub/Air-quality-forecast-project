# db/init.py
import logging
import os
from contextlib import contextmanager
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")


# ─── CONNEXION ───────────────────────────────────────────────────────────────

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session():
    """Context manager pour une session SQLAlchemy avec gestion auto commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Erreur base de données : {e}")
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Vérifie que la connexion à la base de données fonctionne."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        logger.info("Connexion base de données OK")
        return True
    except Exception as e:
        logger.error(f"Connexion base de données FAILED : {e}")
        return False


# ─── STATIONS ────────────────────────────────────────────────────────────────

def upsert_station(name: str, city: str, latitude: float, longitude: float) -> int:
    """
    Insère ou met à jour une station. Retourne l'id de la station.

    Args:
        name: Nom de la station.
        city: Ville de la station.
        latitude: Latitude WGS84.
        longitude: Longitude WGS84.

    Returns:
        id de la station en base.
    """
    sql = text("""
        INSERT INTO stations (name, city, latitude, longitude, geom)
        VALUES (
            :name, :city, :latitude, :longitude,
            ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)
        )
        ON CONFLICT (name, city) DO UPDATE
            SET latitude  = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                geom      = EXCLUDED.geom
        RETURNING id
    """)
    with get_session() as session:
        result = session.execute(sql, {
            "name": name, "city": city,
            "latitude": latitude, "longitude": longitude
        })
        return result.fetchone()[0]


def get_station_id(name: str, city: str) -> int | None:
    """Retourne l'id d'une station par son nom et sa ville, ou None si absente."""
    sql = text("SELECT id FROM stations WHERE name = :name AND city = :city")
    with get_session() as session:
        result = session.execute(sql, {"name": name, "city": city})
        row = result.fetchone()
        return row[0] if row else None


# ─── MESURES ─────────────────────────────────────────────────────────────────

def insert_measurement(
    station_id: int,
    timestamp_utc: datetime,
    pm25: float | None = None,
    no2: float | None = None,
    temperature: float | None = None,
    humidity: float | None = None,
    wind_speed: float | None = None,
    pressure: float | None = None,
) -> bool:
    """
    Insère une mesure brute. Ignore si (station_id, timestamp_utc) existe déjà.

    Returns:
        True si insertion réussie, False si doublon ignoré.
    """
    sql = text("""
        INSERT INTO measurements_raw
            (station_id, timestamp_utc, pm25, no2, temperature, humidity, wind_speed, pressure)
        VALUES
            (:station_id, :timestamp_utc, :pm25, :no2, :temperature, :humidity, :wind_speed, :pressure)
        ON CONFLICT (station_id, timestamp_utc) DO NOTHING
        RETURNING id
    """)
    with get_session() as session:
        result = session.execute(sql, {
            "station_id": station_id,
            "timestamp_utc": timestamp_utc,
            "pm25": pm25, "no2": no2,
            "temperature": temperature, "humidity": humidity,
            "wind_speed": wind_speed, "pressure": pressure,
        })
        return result.fetchone() is not None


def get_latest_measurements(city: str, hours: int = 48) -> list[dict]:
    """
    Retourne les dernières mesures pour une ville sur les N dernières heures.

    Args:
        city: Nom de la ville.
        hours: Nombre d'heures à récupérer.

    Returns:
        Liste de dicts avec les mesures.
    """
    sql = text("""
        SELECT
            m.timestamp_utc, s.name, s.city,
            m.pm25, m.no2, m.temperature, m.humidity, m.wind_speed
        FROM measurements_raw m
        JOIN stations s ON s.id = m.station_id
        WHERE s.city = :city
          AND m.timestamp_utc >= NOW() - INTERVAL ':hours hours'
        ORDER BY m.timestamp_utc DESC
    """)
    with get_session() as session:
        result = session.execute(sql, {"city": city, "hours": hours})
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]


# ─── PRÉDICTIONS ─────────────────────────────────────────────────────────────

def insert_prediction(
    station_id: int,
    predicted_at: datetime,
    horizon: int,
    pm25_pred: float | None = None,
    no2_pred: float | None = None,
    model_name: str = "xgboost",
    model_version: str = "1",
) -> None:
    """
    Insère une prédiction en base.

    Args:
        station_id: Id de la station.
        predicted_at: Timestamp de la prédiction cible.
        horizon: Horizon en heures (ex: 24 pour J+1, 72 pour J+3).
        pm25_pred: Valeur prédite PM2.5.
        no2_pred: Valeur prédite NO2.
        model_name: Nom du modèle utilisé.
        model_version: Version du modèle.
    """
    sql = text("""
        INSERT INTO predictions
            (station_id, predicted_at, horizon, pm25_pred, no2_pred, model_name, model_version)
        VALUES
            (:station_id, :predicted_at, :horizon, :pm25_pred, :no2_pred, :model_name, :model_version)
    """)
    with get_session() as session:
        session.execute(sql, {
            "station_id": station_id,
            "predicted_at": predicted_at,
            "horizon": horizon,
            "pm25_pred": pm25_pred,
            "no2_pred": no2_pred,
            "model_name": model_name,
            "model_version": model_version,
        })


# ─── TEST STANDALONE ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test de connexion...")
    ok = test_connection()

    if ok:
        print("Insertion station test...")
        station_id = upsert_station(
            name="Station Test Paris",
            city="Paris",
            latitude=48.8566,
            longitude=2.3522,
        )
        print(f"Station insérée avec id={station_id}")

        print("Insertion mesure test...")
        inserted = insert_measurement(
            station_id=station_id,
            timestamp_utc=datetime.utcnow(),
            pm25=12.5,
            no2=34.2,
            temperature=18.0,
            humidity=65.0,
            wind_speed=3.2,
            pressure=1013.0,
        )
        print(f"Mesure insérée : {inserted}")
        print("Tout OK.")
