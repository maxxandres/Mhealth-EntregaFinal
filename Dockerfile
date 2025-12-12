# Dockerfile (raíz del proyecto)
FROM python:3.11-slim

# Evitar archivos .pyc y usar stdout sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# 1) Copiamos requirements del backend e instalamos dependencias
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 2) Copiamos código del backend
COPY backend/app /app/app
COPY backend/data /app/data

# 3) Copiamos también el frontend dentro de la imagen
COPY frontend /app/frontend

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando por defecto: levantar la API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
