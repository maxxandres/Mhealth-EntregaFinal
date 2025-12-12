### Proyecto Final – Taller Integrado de Ciencia de Datos  
**Tecnologías:** Python + FastAPI + Vue.js + Docker  
**Dataset:** MHEALTH (UCI Machine Learning Repository)

---

## Descripción General

Este proyecto implementa un **sistema completo de reconocimiento de actividad humana** utilizando el dataset **MHEALTH**, incorporando:

- **Back End (API)** en FastAPI  
- **Modelo predictivo** previamente entrenado (RandomForestClassifier)  
- **Pipeline generalizable** para procesar archivos `.log` del dataset  
- **Front End** interactivo en HTML/CSS/JS con Vue  


---

## Ejecución del Sistema


## Opción A : sin Docker
-- 1. Instalar dependencias

    cd backend
    pip install -r requirements.txt

-- 2. Iniciar el backend

    uvicorn app.main:app --reload

    Para visualizar el backend en el navegador (Gets y Post):

    http://localhost:8000/docs

-- 3. Abrir el frontend

    http://localhost:8000
    

## Opción B: Con Docker:


-- 1. Construir los contenedores

    docker-compose build

-- 2. Ejecutar la aplicación

    docker-compose up

    Luego , la pagina estará corriendo en:

    http://localhost:8000
    



 ## Endpoints Principales   

-- 1. GET /health

    Verifica que el backend esté operativo.

    Respuesta esperada:

    {"status": "ok"}

-- 2. POST /detect

    Procesa un archivo .log y entrega:

        Actividad predominante

        Distribución porcentual

        Tiempo aproximado por actividad

        Detección de anomalía

        Número de filas evaluadas

-- 3. POST /ecg_plot

        Genera y retorna un gráfico PNG del electrocardiograma del archivo enviado.

## 4. Descripción del Pipeline

    El archivo pipeline.py se encarga de:

        Leer el archivo .log

        Estandarizar columnas igual que en el entrenamiento

        Aplicar ventanas temporales (sliding window)

        Preprocesar los datos

        Generar predicciones con el modelo entrenado

        Calcular proporciones, tiempos y anomalías

    Este pipeline garantiza compatibilidad entre entrenamiento y predicción.

#  5. Carpeta prompts

--  La carpeta contiene archivos .md con:

    Los prompts de IA utilizados

    La etapa del proyecto donde fueron aplicados
