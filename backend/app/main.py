# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .pipeline import predict_from_log

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

# Servir frontend estático
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Redirige a la página del frontend si existe.
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
