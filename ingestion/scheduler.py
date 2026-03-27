# ingestion/scheduler.py
import logging
import os
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ingestion.fetch_meteo import run_meteo_ingestion
from ingestion.fetch_openaq import run_ingestion as run_openaq_ingestion

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", "60"))


# ─── JOBS ────────────────────────────────────────────────────────────────────

def job_pipeline() -> None:
    """
    Job principal : lance l'ingestion OpenAQ puis Open-Meteo.
    Exécuté toutes les INTERVAL_MINUTES minutes.
    """
    logger.info(f"=== Pipeline démarré à {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ===")

    try:
        logger.info("--- Étape 1/2 : ingestion OpenAQ ---")
        run_openaq_ingestion()
    except Exception as e:
        logger.error(f"Erreur ingestion OpenAQ : {e}")

    try:
        logger.info("--- Étape 2/2 : ingestion Open-Meteo ---")
        run_meteo_ingestion()
    except Exception as e:
        logger.error(f"Erreur ingestion Open-Meteo : {e}")

    logger.info(f"=== Pipeline terminé à {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ===")


# ─── SCHEDULER ───────────────────────────────────────────────────────────────

def run_scheduler() -> None:
    """
    Lance le scheduler APScheduler.
    - Exécute le pipeline une première fois au démarrage
    - Puis toutes les INTERVAL_MINUTES minutes
    """
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        func=job_pipeline,
        trigger=IntervalTrigger(minutes=INTERVAL_MINUTES),
        id="pipeline_ingestion",
        name="Pipeline OpenAQ + Open-Meteo",
        replace_existing=True,
    )

    logger.info(f"Scheduler démarré — pipeline toutes les {INTERVAL_MINUTES} minutes")
    logger.info("Première exécution immédiate...")

    # Lancer une première fois immédiatement sans attendre
    job_pipeline()

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler arrêté manuellement (Ctrl+C)")
        scheduler.shutdown()


# ─── STANDALONE ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_scheduler()