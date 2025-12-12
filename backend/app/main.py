# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import io
import matplotlib
matplotlib.use("Agg")  # importante en entornos sin display (Docker)
import matplotlib.pyplot as plt

from .pipeline import predict_from_log, load_log_bytes

app = FastAPI(
    title="MHEALTH HAR API",
    description="API para reconocimiento de actividad humana usando el dataset MHEALTH",
    version="1.0.0",
)

# CORS (por si levantas el front por separado)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes restringir luego
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Servir frontend estático
# main.py en Docker queda en /app/app/main.py
# /app/app/main.py -> parent -> /app -> /app/frontend (donde lo copiamos en el Dockerfile)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
# equivalentes: Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Devuelve la página del frontend si existe.
    """
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>MHEALTH HAR API</h1><p>Frontend no encontrado.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Servicio MHEALTH en ejecución"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename.endswith(".log"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .log")

    file_bytes = await file.read()
    try:
        result = predict_from_log(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando el archivo: {e}")

    return JSONResponse(content={
        "filename": file.filename,
        **result
    })


@app.post("/ecg_plot")
async def ecg_plot(file: UploadFile = File(...)):
    """
    Genera un gráfico PNG del electrocardiograma (ecg_1 y ecg_2)
    usando una muestra inicial del archivo .log.
    """
    if not file.filename.endswith(".log"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .log")

    file_bytes = await file.read()
    try:
        df = load_log_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando el archivo: {e}")

    # Tomamos solo una parte para que el gráfico sea manejable
    max_points = 1000
    ecg1 = df["ecg_1"].values[:max_points]
    ecg2 = df["ecg_2"].values[:max_points]
    x = range(len(ecg1))

    # Crear figura con matplotlib
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, ecg1, label="ECG 1")
    ax.plot(x, ecg2, label="ECG 2", alpha=0.7)
    ax.set_title("Electrocardiograma – muestra inicial")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Amplitud")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
