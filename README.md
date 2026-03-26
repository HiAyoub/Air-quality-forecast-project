# 🌫️ air-quality-forecast

> Prédiction de la qualité de l'air urbain en France — pipeline data end-to-end en production

![CI](https://img.shields.io/github/actions/workflow/status/HiAyoub/air-quality-forecast/ci.yml?label=CI&style=flat-square)
![Coverage](https://img.shields.io/badge/coverage--%25-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Problématique

Les pics de pollution urbaine (PM2.5, NO2) sont difficilement anticipables par les citoyens et les collectivités. Ce projet construit un système de prédiction à horizon **J+1 et J+3** pour les grandes villes françaises, en combinant des mesures de pollution temps réel et des données météorologiques.

---

## Architecture

```
OpenAQ API ──┐
              ├──► ETL Python ──► PostgreSQL/PostGIS ──► ML Models ──► FastAPI ──► Streamlit Dashboard
Open-Meteo ──┘                                            (XGBoost │ LSTM)
```

---

## Stack technique

| Couche | Technologie |
|---|---|
| Ingestion | `Python 3.11` · `requests` · `APScheduler` |
| Stockage | `PostgreSQL 15` · `PostGIS 3.4` · `SQLAlchemy` |
| ML | `XGBoost` · `PyTorch (LSTM)` · `scikit-learn` |
| MLOps | `MLflow` (tracking + model registry) |
| API | `FastAPI` · `Pydantic` |
| Dashboard | `Streamlit` · `Plotly` · `Folium` |
| DevOps | `Docker` · `GitHub Actions` · `GCP Cloud Run` |

---

## Résultats

> *Section à compléter après la phase de modélisation*

| Modèle | Horizon | RMSE (PM2.5) | MAE | R² |
|---|---|---|---|---|
| XGBoost | J+1 | — | — | — |
| LSTM | J+1 | — | — | — |
| XGBoost | J+3 | — | — | — |
| LSTM | J+3 | — | — | — |

---

## Lancer en local

```bash
# 1. Cloner le repo
git clone https://github.com/HiAyoub/air-quality-forecast.git
cd air-quality-forecast

# 2. Copier et remplir les variables d'environnement
cp .env.example .env

# 3. Lancer la base de données
docker-compose up -d postgres

# 4. Initialiser le schéma
psql $DATABASE_URL -f db/schema.sql

# 5. Lancer l'ingestion
python ingestion/scheduler.py
```

---

## Démo live

> *Lien à ajouter après déploiement Streamlit Cloud*

---

## Structure du projet

```
air-quality-forecast/
├── ingestion/       # Collecte APIs OpenAQ + Open-Meteo
├── processing/      # Nettoyage, feature engineering, validation
├── db/              # Schéma PostgreSQL + PostGIS, helpers SQLAlchemy
├── models/          # Entraînement XGBoost + LSTM, évaluation, MLflow
├── api/             # FastAPI : /predict /health /cities /history
├── dashboard/       # Streamlit multipage : carte, prédictions, modèles
├── tests/           # pytest : ingestion, processing, API
├── notebooks/       # EDA documentée
├── .github/         # CI/CD GitHub Actions
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Auteur

**Hidara Ayoub** — Data Analyst / Data Engineer
[LinkedIn](https://www.linkedin.com/in/ayoubhidara/) · [GitHub](https://github.com/HiAyoub)