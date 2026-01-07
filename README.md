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

## 🚀 Paso 2.1 Verificar que el workspace quedó creado

```bash
az ml workspace show \
  --name aml-lab-01 \
  --resource-group rg-azureml-lab \
  --query "{name:name,location:location,storage:storage_account}"
```
  ![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/validar_workspace.jpg)

## 🚀 Paso 3 – Crear Compute Instance

- Acceder a Azure ML Studio y crear Compute Instance
Este paso es clave, porque aquí se ejecutan los experimentos.

- Desde el Azure Portal
- Buscar Azure Machine Learning

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/azure_ML.jpg)

- Entra al workspace: aml-lab-01
- También se puede entrar: https://ml.azure.com

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/aml_lab_01.jpg)

- Desde el Azure ML Studio: Clic en Launch Studio
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/launch_studio.jpg)
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/foundry.jpg)


**Crear Instancias de proceso**

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancias_proceso.jpg)

PASO 3.1: Crear la instancia

Instancias de proceso
 Clic en “Agregar instancia de proceso”
 Cuando se abra el formulario:

Nombre:ci-ml-lab

Tipo de máquina

CPU

Tamaño

Standard_DS3_v2


(si no aparece, usa DS2_v2 como alternativa)

Acceso SSH

Desactivado ❌

Apagado por inactividad

✅ Activado

30 minutos

Luego:
➡️ Crear

📌 Qué es una Compute Instance

VM administrada por Azure ML

Ideal para desarrollo, notebooks y pruebas

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso.jpg)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso2.jpg)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso3.jpg)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso4.jpg)

Esperar hasa que termine de crear la instancia y aparezca "Ejecución en curso"
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso5.jpg)

Acceder a la instancia creada:
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/instancia_proceso6.jpg)

## 🚀PASO 4 Crear Notebook ML

En Aplicaciones abrir: JupyterLab
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/jupyterlab.jpg)

Esto abre el Cuaderno:
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/jupyterlab2.jpg)

Crear cuaderno
Seleccionar la instancia
Probar conexión
Ejecutar Python
Cargar dataset
Entrenar modelo (sklearn)

**Crear y ejecutar Notebook en Azure ML**
Crear el Notebook
En JupyterLab:
En el panel izquierdo:
Clic derecho → New Notebook
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook.jpg)

Selecciona:
Kernel: Python 3 (ipykernel)
Nómbralo:proyecto1-azureml.ipynb

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook2.jpg)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook3.jpg)

Probar que el entorno funciona

En la primera celda, pega y ejecuta:
```bash
import sys
print(sys.version)
```

✔️ Debe mostrar la versión de Python
(si corre, todo está perfecto)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook4.jpg)

Ver librerías disponibles

Nueva celda:
```bash
import sklearn
import pandas
import numpy
import matplotlib

print("Todo OK 🚀")
```
👉 En Azure ML las librerías NO siempre vienen preinstaladas
👉 Se instalan por entorno o por notebook

Instalar las librerías directamente en JupyterLab

En una nueva celda, ejecuta EXACTAMENTE esto:

```bash
!pip install -U scikit-learn pandas numpy matplotlib
```
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook5.jpg)

Instalar usando EL MISMO Python del kernel

En una celda nueva, ejecuta exactAMENTE esto:
```bash
import sys
!{sys.executable} -m pip install -U scikit-learn pandas numpy matplotlib
```
sys.executable apunta al Python que usa este notebook.
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook6.jpg)

Reiniciar el kernel (OBLIGATORIO)

En JupyterLab:

Menú Kernel

Restart Kernel

Confirma

⛔ No solo “Restart & Run”, tiene que ser restart limpio.
Verificación

Ejecuta esto en una celda nueva:
```bash
import sklearn
import pandas
import numpy
import matplotlib

print("✅ sklearn disponible, entorno OK")
```

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook7.jpg)
