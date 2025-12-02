"""Train an ANN to predict short-term weather conditions in India.

The script will generate a synthetic dataset (if missing), train a Keras
neural network classifier, and persist both the model and preprocessing
artifacts for reuse by the Flask app.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "synthetic_weather_india.csv"
MODEL_PATH = MODELS_DIR / "weather_ann.keras"
TRANSFORMER_PATH = MODELS_DIR / "feature_transformer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

REGIONS = ["North", "South", "East", "West", "Central"]
CONDITIONS = ["Sunny", "Cloudy", "Rain", "Storm", "Fog"]


def generate_synthetic_row(rng: np.random.Generator) -> dict:
    """Craft a climatologically inspired synthetic observation."""
    region = rng.choice(REGIONS)
    month = int(rng.integers(1, 13))

    # Regional biases (rough heuristics)
    region_temp_bias = {
        "North": -2.5,
        "South": 3.5,
        "East": 1.5,
        "West": 2.0,
        "Central": 0.0,
    }
    region_humidity_bias = {
        "North": -10,
        "South": 8,
        "East": 5,
        "West": -5,
        "Central": 0,
    }
    region_precip_bias = {
        "North": 2.0,
        "South": 5.5,
        "East": 6.0,
        "West": 1.0,
        "Central": 3.5,
    }

    # Seasonal adjustments: (Dec-Feb), (Mar-May), (Jun-Sep), (Oct-Nov)
    if month in (12, 1, 2):
        season = "winter"
    elif month in (3, 4, 5):
        season = "summer"
    elif month in (6, 7, 8, 9):
        season = "monsoon"
    else:
        season = "post_monsoon"

    season_temp_bias = {
        "winter": -7.0,
        "summer": 6.5,
        "monsoon": -1.5,
        "post_monsoon": -0.5,
    }
    season_humidity_bias = {
        "winter": -15,
        "summer": -5,
        "monsoon": 20,
        "post_monsoon": 10,
    }
    season_precip_bias = {
        "winter": 0.5,
        "summer": 1.5,
        "monsoon": 12.0,
        "post_monsoon": 6.0,
    }

    base_temp = 26.0
    temperature_c = base_temp
    temperature_c += region_temp_bias[region]
    temperature_c += season_temp_bias[season]
    temperature_c += rng.normal(0, 3.0)

    humidity_pct = 60.0
    humidity_pct += region_humidity_bias[region]
    humidity_pct += season_humidity_bias[season]
    humidity_pct += rng.normal(0, 7.0)
    humidity_pct = np.clip(humidity_pct, 10, 100)

    precip_mm = 4.0
    precip_mm += region_precip_bias[region]
    precip_mm += season_precip_bias[season]
    precip_mm += max(rng.normal(0, 6.0), -3.0)
    precip_mm = max(0.0, precip_mm)

    cloud_cover_pct = np.clip(precip_mm * rng.uniform(2.5, 4.5) + rng.normal(0, 10), 0, 100)

    pressure_hpa = 1010 + rng.normal(0, 4.5)
    pressure_hpa -= (humidity_pct - 60) * 0.05
    pressure_hpa -= precip_mm * 0.08

    wind_speed_kph = max(1.0, rng.normal(10, 4) + (precip_mm / 6))

    # Determine condition heuristically
    if precip_mm > 35 or (precip_mm > 25 and wind_speed_kph > 18):
        condition = "Storm"
    elif precip_mm > 12 and humidity_pct > 70:
        condition = "Rain"
    elif humidity_pct > 85 and temperature_c < 20 and cloud_cover_pct > 60:
        condition = "Fog"
    elif cloud_cover_pct > 70 and humidity_pct > 65:
        condition = "Cloudy"
    elif humidity_pct < 45 and temperature_c > 32 and precip_mm < 6:
        condition = "Sunny"
    else:
        # Default bucket, nudge toward prevailing pattern
        if season == "monsoon":
            condition = rng.choice(["Rain", "Cloudy"], p=[0.6, 0.4])
        elif season == "winter" and region == "North":
            condition = rng.choice(["Fog", "Sunny"], p=[0.65, 0.35])
        else:
            condition = rng.choice(["Cloudy", "Sunny"], p=[0.55, 0.45])

    return {
        "region": region,
        "month": month,
        "temperature_c": round(float(temperature_c), 2),
        "humidity_pct": round(float(humidity_pct), 2),
        "pressure_hpa": round(float(pressure_hpa), 2),
        "wind_speed_kph": round(float(wind_speed_kph), 2),
        "precip_mm": round(float(precip_mm), 2),
        "cloud_cover_pct": round(float(cloud_cover_pct), 2),
        "condition": condition,
    }


def generate_synthetic_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic weather dataset."""
    rng = np.random.default_rng(seed)
    records = [generate_synthetic_row(rng) for _ in range(rows)]
    df = pd.DataFrame.from_records(records)
    return df


def ensure_dataset() -> pd.DataFrame:
    """Load an existing dataset or create one from scratch."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_synthetic_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Synthetic dataset generated at {DATA_PATH}")

    return df


def build_transformer() -> ColumnTransformer:
    """Construct the preprocessing pipeline for the feature space."""
    categorical_features = ["region"]
    numeric_features = [
        "month",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_kph",
        "precip_mm",
        "cloud_cover_pct",
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("region", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    return transformer


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """Create a Keras Sequential model for classification."""
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train() -> Tuple[keras.Model, ColumnTransformer, LabelEncoder]:
    df = ensure_dataset()

    feature_cols = [
        "region",
        "month",
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_kph",
        "precip_mm",
        "cloud_cover_pct",
    ]
    target_col = "condition"

    X = df[feature_cols]
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    transformer = build_transformer()
    X_transformed = transformer.fit_transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = X_transformed.astype(np.float32)

    y_categorical = keras.utils.to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = build_model(input_dim=X_transformed.shape[1], num_classes=y_categorical.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks,
    )

    val_accuracy = history.history.get("val_accuracy", [float("nan")])[-1]
    print(
        "Training complete. Final metrics:"
        f" loss={history.history['loss'][-1]:.4f}"
        f", val_loss={history.history['val_loss'][-1]:.4f}"
        f", val_accuracy={val_accuracy:.4f}"
    )

    return model, transformer, label_encoder


def persist_artifacts(model: keras.Model, transformer: ColumnTransformer, label_encoder: LabelEncoder) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH, include_optimizer=False)
    joblib.dump(transformer, TRANSFORMER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature transformer saved to {TRANSFORMER_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")


def main() -> None:
    model, transformer, label_encoder = train()
    persist_artifacts(model, transformer, label_encoder)


if __name__ == "__main__":
    main()
