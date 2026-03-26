-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;

-- Table des stations de mesure
CREATE TABLE IF NOT EXISTS stations (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    city        VARCHAR(100) NOT NULL,
    country     VARCHAR(10) DEFAULT 'FR',
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    geom        GEOMETRY(POINT, 4326),
    source      VARCHAR(50) DEFAULT 'openaq',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Table des mesures brutes (une ligne par mesure horaire par station)
CREATE TABLE IF NOT EXISTS measurements_raw (
    id              SERIAL PRIMARY KEY,
    station_id      INTEGER REFERENCES stations(id),
    timestamp_utc   TIMESTAMP NOT NULL,
    pm25            DOUBLE PRECISION,
    no2             DOUBLE PRECISION,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    wind_speed      DOUBLE PRECISION,
    pressure        DOUBLE PRECISION,
    is_outlier      BOOLEAN DEFAULT FALSE,
    is_imputed      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (station_id, timestamp_utc)
);

-- Table des prédictions
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    station_id      INTEGER REFERENCES stations(id),
    predicted_at    TIMESTAMP NOT NULL,
    horizon         INTEGER NOT NULL,
    pm25_pred       DOUBLE PRECISION,
    no2_pred        DOUBLE PRECISION,
    model_name      VARCHAR(100),
    model_version   VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Index pour les performances
CREATE INDEX IF NOT EXISTS idx_measurements_station_time
    ON measurements_raw(station_id, timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_time
    ON measurements_raw(timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_stations_geom
    ON stations USING GIST(geom);

CREATE INDEX IF NOT EXISTS idx_predictions_station_time
    ON predictions(station_id, predicted_at DESC);