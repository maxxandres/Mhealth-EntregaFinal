# backend/app/pipeline.py

from pathlib import Path
import io
import pandas as pd
import numpy as np
import joblib

# Mismas columnas que en el entrenamiento
COLUMNAS = [
    'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
    'ecg_1', 'ecg_2',
    'acc_left_ankle_x', 'acc_left_ankle_y', 'acc_left_ankle_z',
    'gyro_left_ankle_x', 'gyro_left_ankle_y', 'gyro_left_ankle_z',
    'mag_left_ankle_x', 'mag_left_ankle_y', 'mag_left_ankle_z',
    'acc_right_arm_x', 'acc_right_arm_y', 'acc_right_arm_z',
    'gyro_right_arm_x', 'gyro_right_arm_y', 'gyro_right_arm_z',
    'mag_right_arm_x', 'mag_right_arm_y', 'mag_right_arm_z',
    'activity'
]

FEATURE_COLUMNS = COLUMNAS[:-1]  # todo menos 'activity'

# Mapeo de ID de actividad a descripción (según UCI MHEALTH: 1–12) :contentReference[oaicite:0]{index=0}
ACTIVITY_MAP = {
    0: "Null / sin actividad definida",
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back",
}

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"

SCALER_PATH = ML_DIR / "scaler.pkl"
MODEL_PATH = ML_DIR / "rf_model.pkl"

# Carga global para no estar leyendo en cada request
scaler = joblib.load(SCALER_PATH)
rf_model = joblib.load(MODEL_PATH)


def load_log_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Lee un archivo .log de MHEALTH desde bytes y asigna columnas.
    Espera 24 columnas (23 features + label), pero la columna 'activity'
    se puede ignorar en predicción.
    """
    df = pd.read_csv(io.StringIO(file_bytes.decode("utf-8")),
                     sep="\t", header=None)

    if df.shape[1] != 24:
        raise ValueError(f"Se esperaban 24 columnas en el .log, se obtuvieron {df.shape[1]}")

    df.columns = COLUMNAS
    return df


def preprocess_for_model(df: pd.DataFrame):
    """
    Selecciona columnas de features y aplica el scaler entrenado.
    """
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    return X_scaled


def predict_from_log(file_bytes: bytes):
    """
    Pipeline completo: lee el .log, lo transforma y obtiene predicciones.
    Devuelve resumen y distribución de actividades.
    """
    df = load_log_bytes(file_bytes)
    X_scaled = preprocess_for_model(df)

    preds = rf_model.predict(X_scaled)
    preds = preds.astype(int)

    # actividad dominante (la más frecuente en la ventana completa)
    main_activity_id = int(pd.Series(preds).mode()[0])
    main_activity_name = ACTIVITY_MAP.get(main_activity_id, "Actividad desconocida")

    # distribución (porcentaje de filas por actividad)
    value_counts = pd.Series(preds).value_counts(normalize=True).sort_index()
    distribution = {
        int(k): float(v)
        for k, v in value_counts.round(3).to_dict().items()
    }

    # criterio simple de "anomalía":
    # si la clase 0 (null) domina, lo marcamos como anómalo
    is_anomaly = distribution.get(0, 0.0) > 0.5

    return {
        "predicted_activity_id": main_activity_id,
        "predicted_activity_name": main_activity_name,
        "activity_distribution": distribution,
        "rows_evaluated": len(preds),
        "is_anomaly": is_anomaly,
    }
