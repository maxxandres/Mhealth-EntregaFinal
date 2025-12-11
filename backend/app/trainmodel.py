# backend/app/train_model.py

import os
import glob
import io
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path

DATA_URL = "https://download1527.mediafire.com/0ev6a94ewbcgKBrZAnvAm3tX8skvHkjoC30i5gMya8kRUC20hwnHxJOvUP92AG4vHENEIwJzEpuug355DiTeguUaRrkc_FPSFclCJJryu_LrCOUOi0uDC6oG67vMSmDH734HSXNqS_vHfLhkTAxmXgFsRRKX0ZZSYQ04k7WpqHSENQ/b5idi3b8lgou39c/mhealth%2Bdataset.zip"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "MHEALTHDATASET"
ML_DIR = Path(__file__).resolve().parent / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

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


def download_mhealth():
    if RAW_DIR.exists():
        print(f"[OK] Datos ya existen en {RAW_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("[*] Descargando dataset MHEALTH...")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(DATA_DIR)

    print(f"[OK] Dataset extraído en {DATA_DIR}")


def load_full_dataframe():
    archivos = glob.glob(str(RAW_DIR / "*.log"))
    if not archivos:
        raise RuntimeError(f"No se encontraron .log en {RAW_DIR}. ¿Descargaste bien el dataset?")

    dfs = []
    for archivo in archivos:
        df_temp = pd.read_csv(archivo, sep="\t", header=None)
        df_temp["Subject"] = os.path.basename(archivo).replace(".log", "")
        dfs.append(df_temp)

    df_total = pd.concat(dfs, ignore_index=True)
    df_total.columns = COLUMNAS + ["Subject"]
    print("[OK] Datos combinados:", df_total.shape)
    return df_total


def train_and_save_model():
    download_mhealth()
    df_total = load_full_dataframe()

    X = df_total.drop(['activity', 'Subject'], axis=1)
    y = df_total['activity']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    print("[*] Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[RESULTADOS] Accuracy: {acc:.3f} | F1: {f1:.3f}")

    joblib.dump(scaler, ML_DIR / "scaler.pkl")
    joblib.dump(rf_model, ML_DIR / "rf_model.pkl")
    print(f"[OK] Modelos guardados en {ML_DIR}")


if __name__ == "__main__":
    train_and_save_model()
