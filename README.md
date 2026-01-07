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
- Documentar el proceso para uso profesional

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
- Ingresar a Portal de azure
- Tener habilidatada una suscripcion de azure
- Cloud Shell
- Arriba seleccionar Bash (no PowerShell)
![Cloud Shell](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/cloudshell.jpg)

Ejecutar los comandos:
```bash
az login
az group create \
  --name rg-azureml-lab \
  --location eastus
```

  ![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/group_create.jpg)


```

```
Un Resource Group es un contenedor lógico

Permite administrar costos, permisos y borrado

## 🚀 Paso 2 – Crear Azure ML Workspace en Europa (Bash)

👉 Asegúrate de estar en Cloud Shell – Bash

Ejecutar los comandos:
```bash

az ml workspace create \
  --name aml-lab-eu \
  --resource-group rg-azureml-lab \
  --location westeurope
```

⏳ Esperar 2–5 minutos.

Azure crea automáticamente:

- Storage Account
- Key Vault- 
- Applicati- on Insights
- Container-  Registry (opciona- l)

  ![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/ML_workspace.jpg)

Verificar que el workspace quedó creado


az ml workspace show \
  --name aml-lab-01 \
  --resource-group rg-azureml-lab \
  --query "{name:name,location:location,storage:storage_account}"

  ![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/validar_workspace.jpg)

## 🚀 Paso 3 – Crear Compute Instance

Desde el Azure ML Studio:

Ir a Compute

Crear Compute Instance

Tamaño recomendado:

Standard_DS11_v2

Sistema: Ubuntu

📌 Qué es una Compute Instance

VM administrada por Azure ML

Ideal para desarrollo, notebooks y pruebas

  ![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/ML_workspace.jpg)
