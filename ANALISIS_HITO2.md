# Análisis de Resultados — WikiArt Hito 2

**Proyecto:** CC5205 Minería de Datos — Grupo 6, Sección 3  
**Dataset:** 76.5k pinturas etiquetadas por género, restringido a las 10 clases más frecuentes (61.208 obras para clasificación, 12.242 en test).

---

## 📁 1. Estructura del Repositorio

A continuación se detalla la estructura actual del repositorio tras la consolidación de cambios:

| Archivo / Directorio | Descripción |
| :--- | :--- |
| [wikiart_hito1.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/hito1/wikiart_hito1.ipynb) | Notebook del Hito 1 (EDA + modelos baseline tabulares). |
| [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb) | Notebook unificado del Hito 2 (balanceo de clases, multilabel, ResNet, clustering, Grad-CAM). |
| [requirements.txt](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/requirements.txt) | Dependencias del proyecto (incluye `opencv-python` e `imbalanced-learn`). |
| [CSVs/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/CSVs) | Archivos de metadatos y características geométricas (`classes.csv`, `visual_features.csv`, `wclasses.csv`). |
| [images/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/images) | Directorio con las **81,444 imágenes reales** clasificadas por género (excluido de Git mediante `.gitignore`). |

---

## 📊 2. Análisis del Baseline (Hito 1) — Random Forest Tabular

### Distribución de Géneros (Top-10)
El dataset presenta un **ratio de desbalance de 5.1×** entre la clase mayoritaria (*Impressionism*) y la minoritaria (*Northern Renaissance*):

| Rank | Género | # Pinturas |
| :---: | :--- | :---: |
| 1 | Impressionism | 13,028 |
| 2 | Realism | 10,546 |
| 3 | Romanticism | 6,919 |
| 4 | Expressionism | 6,335 |
| 5 | Post Impressionism | 6,307 |
| 6 | Symbolism | 4,524 |
| 7 | Baroque | 4,236 |
| 8 | Art Nouveau Modern | 4,168 |
| 9 | Abstract Expressionism | 2,594 |
| 10 | Northern Renaissance | 2,551 |

### Resultados del Baseline (Sin Balancear)
* **Accuracy:** 0.24
* **F1 Macro:** 0.13
* **F1 Weighted:** 0.18

### Análisis de Colapso de Clases (Baseline)
* **Impressionism:** Precision 0.27, Recall **0.66**, F1 0.38 (Atrae 2/3 de las predicciones).
* **Realism:** Precision 0.23, Recall **0.42**, F1 0.29 (Segundo sumidero de predicciones).
* **Clases minoritarias (ej. Northern Renaissance):** Recall **0.02**, F1 0.04 (Prácticamente ignoradas).

> [!IMPORTANT]
> **Diagnóstico:** Las features tabulares utilizadas son puramente **geométricas** (`aspect_ratio`, `log_width`, `log_height`, `is_portrait`, `is_square`, `es_multilabel`). Al no contener información visual o de estilo, el modelo opta por predecir las clases mayoritarias para maximizar su acierto general.

---

## ⚖️ 3. Estrategias de Balanceo de Clases (Hito 2)

Se compararon tres técnicas de balanceo sobre el mismo conjunto de test desbalanceado:

### Resumen Comparativo (Métricas Globales)

| Modelo | Accuracy | F1 Macro | F1 Weighted |
| :--- | :---: | :---: | :---: |
| **Baseline (Sin balancear)** | **0.24** | 0.13 | 0.18 |
| Class Weighting (`class_weight='balanced'`) | 0.15 | 0.15 | 0.15 |
| **Undersampling Manual** (a la mediana) | 0.23 | **0.20** | **0.22** |
| Balanced Random Forest (`imblearn`) | 0.15 | 0.15 | 0.15 |

### Hallazgos Clave

* **El undersampling manual a la mediana fue la mejor técnica.**
  Mejora simultáneamente el F1-macro (0.13 → 0.20) y F1-weighted (0.18 → 0.22) sin destruir el accuracy global (0.24 → 0.23). Al recortar las clases mayoritarias hasta la mediana (4,524 muestras) en lugar del mínimo, preserva un conjunto de datos grande (45,148 muestras) reduciendo la pérdida de información.
* **Class Weighting y Balanced Random Forest convergen al mismo límite.**
  Ambas técnicas reportan métricas idénticas (F1-macro 0.15 / Accuracy 0.15). En un espacio de características geométricas sin poder discriminativo real, modificar los pesos de las clases a nivel de impureza (`class_weight`) o a nivel de bootstrap (`BalancedRF`) solo consigue que el modelo arriesgue predicciones hacia las minoritarias sin una base real, aumentando su recall pero colapsando su precisión.

---

## 🏷️ 4. Clasificación Multietiqueta (Multilabel)

* **Estrategia:** OneVsRest + Random Forest (50 árboles, profundidad 10) sobre 27 clases binarias obtenidas de `genre_list`.
* **Métricas macro/micro/weighted F1:** **0.01**

> [!WARNING]
> El clasificador devuelve casi siempre predicciones negativas en todas las clases. Las features geométricas y el umbral por defecto de 0.5 impiden que clases de muy baja prevalencia superen la probabilidad crítica de asignación. Para el Hito 3, será clave evaluar con métricas de probabilidad (`average_precision_score`) o calibrar umbrales por clase en lugar del 0.5 genérico.

---

## 🧠 5. Deep Learning: Embeddings de ResNet-18 y Clustering

### Configuración del Pipeline
* **Modelo:** ResNet-18 preentrenada en ImageNet sin fine-tuning (usada como extractor de características).
* **Dimensión de Embeddings:** 512
* **Tamaño del Muestreo:** `N_SAMPLES = 2974`

### Resultados de la Extracción Real (CPU)
* **Tiempo de Extracción:** ~3 minutos y 40 segundos (a un promedio de 2.38s/it).
* **Coeficiente Silhouette obtenido:** **0.079** (K=4 clusters tras reducción PCA a 50 componentes).

> [!NOTE]
> **Análisis del Coeficiente Silhouette (0.079 vs 0.751 del Hito 1):**
> En el Hito 1, el clustering sobre features manuales obtuvo un Silhouette score de **0.751**. Sin embargo, esto era un **espejismo**: las features eran geométricas y estructuraban los datos en grupos densos basados en la orientación física del lienzo (ej. retrato vs vertical vs cuadrado).
>
> En el Hito 2, al usar embeddings de ResNet-18, agrupamos por características **visuales y semánticas reales** de la pintura. El coeficiente **0.079** es bajo pero biológicamente y artísticamente correcto: los estilos pictóricos no están aislados en cajas compactas, sino que forman un continuo estilístico con alta superposición en el espacio visual latente.

---

## 🔍 6. Interpretabilidad Visual con Grad-CAM

* **Capa objetivo:** `resnet.layer4[-1]` (último bloque convolucional).
* **Resultado:** Se ha generado el mapa de calor exitosamente sobre una obra real del DataLoader.
* **Análisis cualitativo:** Al usar una ResNet-18 preentrenada en ImageNet, las activaciones se centran en formas geométricas y contornos de objetos generales (ej. bordes de objetos prominentes) más que en características estilísticas específicas (como pinceladas o texturas de óleo). 

---

## 🚀 7. Plan de Trabajo Futuro: Hito 2 a Hito 3

Una vez que el notebook funciona de extremo a extremo con imágenes reales, el grupo se encuentra en una excelente posición para desarrollar las siguientes extensiones del proyecto:

### 1. Clasificación usando Embeddings Visuales (Hito 3)
* **Problema actual:** Los clasificadores de la Sección 2 y 3 usan variables geométricas con un tope de rendimiento del 24% de accuracy.
* **Mejora:** Entrenar un Random Forest, una Regresión Logística o una SVM usando los **embeddings de 512 dimensiones extraídos por ResNet-18** en lugar de las features geométricas. Esto debería disparar las métricas de clasificación, pues el modelo tendrá acceso a información visual genuina.

### 2. Fine-Tuning de ResNet-18 (Hito 3)
* **Problema actual:** Los embeddings provienen de un extractor entrenado en ImageNet (fotos del mundo real).
* **Mejora:** Descongelar las últimas capas convolucionales de ResNet-18 y entrenar la red directamente sobre el dataset de WikiArt para clasificar géneros artísticos. Esto forzará al modelo a aprender características pictóricas específicas (pinceladas, trazos, paletas de colores).
* **Impacto en Grad-CAM:** Los mapas de calor de Grad-CAM se volverán mucho más interesantes e interpretables, resaltando las texturas y pinceladas que definen cada movimiento en lugar de objetos genéricos.

### 3. Ajuste de Umbrales Multietiqueta
* **Problema actual:** El umbral por defecto (0.5) colapsa la clasificación multietiqueta.
* **Mejora:** Utilizar `predict_proba` y buscar un umbral optimizado por clase (ej. el cuantil de probabilidad en la partición de validación) para recuperar el recall de clases minoritarias en el esquema multietiqueta.
