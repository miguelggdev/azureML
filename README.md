# Proyecto – Entrenamiento de un modelo ML con Azure Machine Learning y scikit-learn

## 📌 Descripción
Este proyecto demuestra el uso de **Azure Machine Learning** para entrenar, evaluar y registrar un modelo de Machine Learning usando **scikit-learn**.

El objetivo es practicar el flujo completo de trabajo de Azure ML, desde la creación del workspace hasta el registro del modelo, siguiendo buenas prácticas profesionales.

---

## 🧠 Objetivos del proyecto
- Crear un Azure ML Workspace
- Configurar Compute Instance
- Entrenar un modelo ML en JupyterLab
- Evaluar métricas del modelo
- Registrar el modelo en Azure ML
- Documentar el proceso para uso profesional y entrevistas

---

## 🛠️ Tecnologías utilizadas
- Azure Machine Learning
- Azure CLI
- Python 3
- scikit-learn
- pandas
- numpy
- JupyterLab

---

## 🧱 Arquitectura
Usuario
├── Azure Portal / Azure ML Studio
│ ├── Azure ML Workspace
│ │ ├── Compute Instance
│ │ ├── JupyterLab
│ │ └── Model Registry
└── Azure CLI


---

## 🚀 Paso 1 – Crear Resource Group
Ingresar a Portal de azure
Tener habilidatada una suscripcion de azure
Abrir Cloud Shell
Arriba seleccionar Bash (no PowerShell)
![Texto alternativo](https://ruta-de-la-imagen.com/imagen.png)

Ejecutar los comandos:
```bash
az login
az group create \
  --name rg-azureml-lab \
  --location eastus



- Imagen local (ruta relativa):

  ![Texto alternativo](screenshots/ejemplo.png)

  - Coloca la imagen dentro de la carpeta `screenshots` (o la que prefieras). En GitHub las rutas relativas funcionan automáticamente.

- Imagen remota (URL completa):

  ![Texto alternativo](https://example.com/imagen.png)

- Controlar tamaño (HTML en MD):

  <img src="screenshots/ejemplo.png" alt="Ejemplo" width="400" />

- Nota: la sintaxis básica de Markdown es `![alt](ruta)` donde `ruta` puede ser relativa o absoluta.

Ejemplo práctico:

```
![Diagrama del flujo](screenshots/diagrama.png)
```

Si quieres, puedo añadir una imagen real (placeholder) en `screenshots/` y enlazarla aquí.


