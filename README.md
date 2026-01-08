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

**🧠 scikit-learn**

scikit-learn (sklearn) es la librería de Machine Learning más usada en Python para:

📊 Análisis de datos

🤖 Entrenamiento de modelos ML

📈 Evaluación de modelos

🧪 Experimentos rápidos y confiables

👉 Es la base del Machine Learning clásico (NO deep learning).

Entrenar modelos de Machine Learning

Permite crear modelos como:

Tipo	Ejemplos
Clasificación	Logistic Regression, Random Forest, SVM
Regresión	Linear Regression, Ridge, Lasso
Clustering	K-Means, DBSCAN
Reducción de dimensión	PCA

🧩 ¿Por qué scikit-learn es tan importante en Azure ML?

Azure ML:

NO reemplaza scikit-learn

LO ORQUESTA

Azure ML se encarga de:

💻 Infraestructura

☁️ Escalado

📦 Versionado

📊 Experimentos

🚀 Despliegue

Mientras que scikit-learn:

Hace el ML real (modelos)

👉 Azure ML + scikit-learn = combo estándar empresarial
Ejemplo simple:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```
Preparar y limpiar datos (MUY importante)

Incluye herramientas para:

Escalar datos

Normalizar

Codificar texto y categorías

Manejar valores nulos

Ejemplo:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Dividir datos correctamente**

Evita errores graves de ML:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Medir qué tan bueno es el modelo

Scikit-learn no solo entrena, también evalúa:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```
Pipelines (nivel profesional)

Permite unir todo el flujo:

from sklearn.pipeline import Pipeline

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

scikit-learn	     Azure ML
Entrena modelos	  Orquesta el proceso
Código ML	      Infraestructura
Funciona solo	  Escala a empresa Cloud
Local	

👉 Azure ML NO reemplaza sklearn
👉 Azure ML profesionaliza sklearn

PASO 5
Entrenar  modelo (Iris)

Copia todo este bloque en una celda nueva y ejecútalo:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy del modelo: {accuracy:.2f}")
```
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook8.jpg)

Qué acabas de hacer (muy importante)

Cargaste un dataset estándar

Entrenaste un modelo real

Evaluaste resultados

Usaste exactamente el mismo flujo que en producción

Resultado esperado

Verás algo como:

Accuracy del modelo: 0.96


(entre 0.90 y 1.00 es normal)

Accuracy = 1.00 significa que el modelo clasificó perfectamente los datos de prueba (algo normal en Iris).

🌸 ¿Qué es IRIS?

El dataset Iris es el “Hola Mundo” del Machine Learning.

📊 Contiene:

150 flores Iris

3 especies:

Iris setosa

Iris versicolor

Iris virginica

4 características (features) por flor:

Largo del sépalo

Ancho del sépalo

Largo del pétalo

Ancho del pétalo

👉 El objetivo del modelo es:

Dado el tamaño de una flor, predecir su especie

🧠 Qué tipo de problema es

✔️ Clasificación supervisada

✔️ Multiclase (3 clases)

✔️ Datos numéricos

✔️ Dataset pequeño y limpio

Por eso es perfecto para aprender.

🔍 EXPLICACIÓN DEL CÓDIGO (línea por línea)
1️⃣ Importaciones
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

```
Qué hace cada una:
Línea	Función
load_iris	Carga el dataset
train_test_split	Divide datos para entrenar y probar
RandomForestClassifier	Algoritmo de ML
accuracy_score	Métrica de evaluación

Cargar los datos

```python
iris = load_iris()
X = iris.data
y = iris.target
```
Qué es iris

Es un objeto con:

data → las características (X)

target → la etiqueta (y)

target_names → nombres de especies

feature_names → nombres de columnas

En ML:

X = lo que el modelo ve

y = lo que el modelo debe aprender a predecir
Dividir los datos
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Por qué se hace:

80% → entrenamiento

20% → prueba (datos nunca vistos)

❗ Esto evita autoengaño (overfitting).

random_state=42

Hace que la división sea reproducible

Fundamental en ciencia de datos

Entrenar el modelo
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
¿Qué es Random Forest?

Un conjunto de árboles de decisión

Cada árbol vota

El modelo decide por mayoría

n_estimators=100

Número de árboles

Más árboles → mejor generalización (hasta cierto punto)

.fit()

👉 Aquí ocurre el aprendizaje

5️⃣ Predicción
```python
y_pred = model.predict(X_test)
```
El modelo ve datos nuevos

Devuelve la clase predicha

6️⃣ Evaluación
```python
accuracy = accuracy_score(y_test, y_pred)
```
Accuracy significa:

Porcentaje de predicciones correctas

1.00 = 100% correcto

👉 En Iris es normal
👉 En problemas reales NO siempre es bueno (alerta de overfitting)

¿Qué APRENDIÓ el modelo?

Aprendió patrones como:

Si pétalos son cortos → Setosa

Si pétalos largos y anchos → Virginica

Combinaciones intermedias → Versicolor

Próximo paso (PASO 4)

Ahora vamos a:
🔥 Registrar este experimento en Azure ML

Experimentos

Métricas

Tracking profesional

Registrar el experimento y métricas en Azure ML

En este paso NO cambiamos el modelo, solo le decimos a Azure:

👉 “Esto es un experimento, guarda métricas, resultados y ejecuciones”

Esto es lo que las empresas usan y revisan.

🧠 Qué vas a aprender en este paso

Qué es un experimento en Azure ML

Cómo loggear métricas

Cómo ver resultados en Azure ML Studio

Diferencia entre código local vs código productivo

1️⃣ Importar Azure ML SDK (v2)

En una nueva celda, ejecuta:
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
```

Instalar Azure ML SDK v2 (en el notebook)

En una celda nueva, ejecuta:

```python
!pip install azure-ai-ml azure-identity

```
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook9.jpg)

PASO 2 – Reiniciar el Kernel (MUY IMPORTANTE)

Esto es clave y muchos lo olvidan.

En JupyterLab:

Arriba → Kernel

Restart Kernel

Confirmar

📌 Sin reiniciar, Python no detecta el nuevo paquete.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

print("Azure ML SDK cargado correctamente")
```

Verifica qué Python está usando el kernel

En una celda, ejecuta:

import sys
sys.executable


Guarda mentalmente la ruta que te salga
(Ejemplo: /anaconda/envs/azureml_py38/bin/python)

Tu kernel está usando este Python:

/anaconda/envs/jupyter_env/bin/python


👉 Eso confirma al 100% que el SDK no está instalado en ese entorno, por eso falla el import.

Ahora vamos directo a la solución correcta, sin rodeos.

✅ PASO ÚNICO (SOLUCIÓN DEFINITIVA)

En una celda nueva del notebook, ejecuta exactamente esto:

```python
!/anaconda/envs/jupyter_env/bin/python -m pip install azure-ai-ml azure-identity
```


⏳ Espera a que termine (puede tardar 1–2 minutos).

🔄 PASO OBLIGATORIO – Reiniciar Kernel

Cuando termine:

Menú Kernel

Restart Kernel

Confirmar

⚠️ Este paso es obligatorio, no opcional.

✅ VERIFICACIÓN FINAL

Después del reinicio, ejecuta:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

print("Azure ML SDK cargado correctamente ✅")
```


Si ves el mensaje → problema resuelto definitivamente.
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook10.jpg)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook11.jpg)

Recordar: 
Esto que te pasó es muy común en Azure ML y es una habilidad real:

Los kernels no comparten paquetes

Azure ML no garantiza que el SDK esté instalado

Usar sys.executable o la ruta directa es best practice

Tip importante

Si no sabes tu subscription_id, ejecútalo en terminal:
az account show --query id -o tsv
PASO 5 – Conectarte al Azure ML Workspace desde Python

En tu notebook, ejecuta una celda nueva:
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="14e40c61-1bae-4b1d-b5a3-efcd3059264a",
    resource_group_name="rg-azureml-lab",
    workspace_name="aml-lab-01"
)

print("Conectado al workspace:", ml_client.workspace_name)

```
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook12.jpg)

🟢 PASO 6 – Guardar el modelo entrenado localmente

Azure ML no registra modelos en memoria, primero hay que guardarlos.

Ejecuta:
```python
import joblib
import os

os.makedirs("outputs", exist_ok=True)

model_path = "outputs/iris_random_forest.pkl"
joblib.dump(model, model_path)

print("Modelo guardado en:", model_path)
```
Esto crea la carpeta outputs/
Azure ML la detecta automáticamente.

Se cerro la Instancia por tiempo de 30 min, hay que volver a correr el modelo para guardarlo: SOLUCIÓN (ORDEN CORRECTO)
🔹 PASO 1 – Reentrenar el modelo (obligatorio)

Ejecuta de nuevo esta celda COMPLETA:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy del modelo: {accuracy:.2f}")


Confirma que vuelve a salir:

Accuracy del modelo: 1.00

🔹 PASO 2 – Guardar el modelo (ahora sí)

Ahora ejecuta:

import joblib
import os

os.makedirs("outputs", exist_ok=True)

model_path = "outputs/iris_random_forest.pkl"
joblib.dump(model, model_path)

print("Modelo guardado en:", model_path)

![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook13.jpg)

**Registrar el modelo en Azure ML**

Después de guardar, ejecuta:

```python
from azure.ai.ml.entities import Model

ml_model = Model(
    path=model_path,
    name="iris-random-forest",
    description="Random Forest trained on Iris dataset",
    type="custom_model"
)

registered_model = ml_client.models.create_or_update(ml_model)

print("Modelo registrado:", registered_model.name)
print("Versión:", registered_model.version)
```
![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook14.jpg)

🧠 Lección importante (nivel PRO)

En notebooks:

El orden de ejecución importa

Reiniciar kernel borra memoria

Azure ML no guarda estados automáticamente

💬 Frase para entrevista:

“En notebooks siempre controlo el orden de ejecución para evitar inconsistencias de estado.”

✅ Lo que lograste (checklist real)

✔️ Creaste Resource Group
✔️ Creaste Azure ML Workspace vía CLI
✔️ Configuraste Compute Instance
✔️ Usaste JupyterLab en Azure
✔️ Entrenaste modelo con scikit-learn
✔️ Evaluaste métricas (Accuracy = 1.00)
✔️ Instalaste y usaste Azure ML SDK v2
✔️ Conectaste al Workspace desde Python
✔️ Guardaste el modelo
✔️ Registraste el modelo con versionado

👉 Esto ya es MLOps básico.

FORMAS DE PROBAR TU MODELO (AZURE ML)

OPCIÓN 1 — Prueba LOCAL desde Jupyter (la más rápida)

👉 Ideal para validar que el modelo funciona y predice bien.

📌 Qué haces

Cargas el modelo registrado

Le pasas datos nuevos

Ves la predicción

🧠 Flujo
Modelo registrado → lo cargo → predigo

🔹 Código (Jupyter)

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import joblib
import pandas as pd

# Conectar al workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="14e40c61-1bae-4b1d-b5a3-efcd3059264a",
    resource_group_name="rg-ml",
    workspace_name="ml-workspace"
)

# Descargar modelo registrado
model = ml_client.models.get(name="iris-rf-model", version="1")
model_path = ml_client.models.download(name=model.name, version=model.version)

# Cargar modelo
rf_model = joblib.load(f"{model_path}/iris_random_forest.pkl")

# Datos de prueba
data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Predicción
prediction = rf_model.predict(data)
print("Predicción:", prediction)

```
✅ OPCIÓN 2 — Batch inference (pruebas con muchos datos)

👉 Simula uso real: CSV completo de pruebas.

📌 Qué haces

Cargas un CSV

El modelo predice todo el dataset

Exportas resultados

🔹 Ejemplo
test_data = pd.read_csv("test_data.csv")
predictions = rf_model.predict(test_data)

test_data["prediction"] = predictions
test_data.to_csv("outputs/predictions.csv", index=False)


✔️ Esto ya es testing de datos reales.

🚀 OPCIÓN 3 — Endpoint REST (nivel producción)

👉 Aquí el modelo se comporta como servicio web.

POST → Endpoint → Modelo → Predicción


Esto es lo que piden en empresas.

📌 Flujo real

Crear endpoint

Desplegar modelo

Probar con curl o Postman

👉 Esto será el Proyecto 2.5, no te lo lanzo todavía para no mezclar conceptos.

📊 MÉTRICAS PARA PRUEBAS (IMPORTANTE)

No basta con “funciona”.

Métricas básicas:

Accuracy

Confusion Matrix

Precision / Recall

Ejemplo rápido:

from sklearn.metrics import classification_report

y_true = y_test
y_pred = rf_model.predict(X_test)

print(classification_report(y_true, y_pred))

🧠 ¿Cómo lo explicas en entrevistas?

“Validé el modelo mediante pruebas locales, batch inference con datasets completos y dejé preparado el modelo para despliegue en endpoints REST en Azure ML.”

🔥🔥🔥

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import joblib
import pandas as pd

# 1️⃣ Conectar al workspace correcto
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="14e40c61-1bae-4b1d-b5a3-efcd3059264a",
    resource_group_name="rg-azureml-lab",
    workspace_name="aml-lab-01"
)

print("✅ Conectado a Azure ML")

# 2️⃣ Obtener modelo registrado
model = ml_client.models.get(
    name="iris-rf-model",
    version="1"
)

# 3️⃣ Descargar modelo
model_path = ml_client.models.download(
    name=model.name,
    version=model.version
)

print("📦 Modelo descargado en:", model_path)

# 4️⃣ Cargar modelo
rf_model = joblib.load(f"{model_path}/iris_random_forest.pkl")

# 5️⃣ Datos de prueba
data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# 6️⃣ Predicción
prediction = rf_model.predict(data)

print("🌸 Predicción del modelo:", prediction)
FORMA PROFESIONAL DE ARREGLARLO (RECOMENDADO)
✔️ Opción A (la mejor para GitHub y entrevistas)

Entrenar el modelo desde el inicio con DataFrame y nombres de columnas

🔁 Cambia el entrenamiento original a esto:
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Cargar datos
iris = load_iris(as_frame=True)
X = iris.data          # DataFrame con nombres
y = iris.target

# 2️⃣ Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy del modelo: {accuracy:.2f}")

```


![Crear Resource Group](https://github.com/miguelggdev/azureML/blob/main/project-01-azureml-sklearn/screenshots/newnotebook15.jpg)