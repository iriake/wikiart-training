# Análisis de Resultados — WikiArt Hito 2

**Proyecto:** CC5205 Minería de Datos — Grupo 6, Sección 3  
**Dataset:** 76.5k pinturas etiquetadas por género, restringido a las 10 clases más frecuentes (61.208 obras para clasificación, 12.242 en test).

---

## 📁 1. Estructura del Repositorio

A continuación se detalla la estructura del repositorio tras la ejecución del notebook y consolidación de archivos:

| Archivo / Directorio | Descripción |
| :--- | :--- |
| [wikiart_hito1.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/hito1/wikiart_hito1.ipynb) | Notebook del Hito 1 (EDA + modelos baseline tabulares). |
| [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb) | Notebook unificado del Hito 2 (balanceo de clases, multilabel, ResNet, clustering, Grad-CAM, clasificación y regresión sobre embeddings). |
| [requirements.txt](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/requirements.txt) | Dependencias actualizadas del proyecto (incluye `opencv-python` e `imbalanced-learn`). |
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
* **F1 Macro:** 0.138 (redondeado en el notebook a 0.13)
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
| **Baseline (Sin balancear)** | **0.24** | 0.138 | 0.18 |
| Class Weighting (`class_weight='balanced'`) | 0.15 | 0.15 | 0.15 |
| **Undersampling Manual** (a la mediana) | 0.23 | **0.20** | **0.22** |
| Balanced Random Forest (`imblearn`) | 0.15 | 0.15 | 0.15 |

### Hallazgos Clave

* **El undersampling manual a la mediana fue la mejor técnica.**
  Mejora simultáneamente el F1-macro (0.138 → 0.20) y F1-weighted (0.18 → 0.22) sin destruir el accuracy global (0.24 → 0.23). Al recortar las clases mayoritarias hasta la mediana (4,524 muestras) en lugar del mínimo, preserva un conjunto de datos grande (45,148 muestras) reduciendo la pérdida de información.
* **Class Weighting y Balanced Random Forest convergen al mismo límite.**
  Ambas técnicas reportan métricas idénticas (F1-macro 0.15 / Accuracy 0.15). En un espacio de características geométricas sin poder discriminativo real, modificar los pesos de las clases a nivel de impureza (`class_weight`) o a nivel de bootstrap (`BalancedRF`) solo consigue que el modelo arriesgue predicciones hacia las minoritarias sin una base real, aumentando su recall pero colapsando su precisión.

---

## 🏷️ 4. Clasificación Multietiqueta (Multilabel)

* **Estrategia:** OneVsRest + Random Forest (50 árboles, profundidad 10) sobre 27 clases binarias obtenidas de `genre_list`.
* **Métricas macro/micro/weighted F1:** **0.01**

> [!WARNING]
> **Nota Metodológica sobre Features Tabulares:**
> Al igual que en la clasificación multiclase del baseline, la clasificación multietiqueta en esta sección utiliza las características tabulares geométricas del lienzo (como `aspect_ratio` y dimensiones). Dado que estas variables carecen de información visual sobre el estilo artístico de las obras, se espera por diseño que el rendimiento (F1-score) sea sumamente bajo.
> 
> El clasificador devuelve casi siempre predicciones negativas en todas las clases. El objetivo primordial de esta sección es **implementar y validar la infraestructura del pipeline multietiqueta** (Binarización de etiquetas múltiples, clasificador One-Vs-Rest y matrices de confusión multietiqueta), sirviendo como base estructural para futuros experimentos con características visuales profundas en el Hito 3.

---

## 🧠 5. Deep Learning: Embeddings de ResNet-18 y Clustering

### Configuración del Pipeline
* **Modelo:** ResNet-18 preentrenada en ImageNet sin fine-tuning (usada como extractor de características).
* **Dimensión de Embeddings:** 512
* **Tamaño del Muestreo:** **`N_SAMPLES = 10000`** (escala de 10k+ para cumplir con la propuesta del Hito 1).

### Resultados de la Extracción Real (GPU / CUDA)
* **Hardware Utilizado:** NVIDIA GeForce GTX 1050 Ti (CUDA 12.6).
* **Tiempo de Extracción:** ~7 minutos y 46 segundos (a un promedio de 1.49s/it para las 10,000 imágenes).
* **Sweep de K para Clustering (K=2 a 8):**
  * K=2: 0.091
  * K=3: 0.089
  * **K=4: 0.079** (usado para consistencia con Hito 1)
  * K=5: 0.079
  * K=6: 0.067
  * K=7: 0.074
  * K=8: 0.075

### Comparación Cuantitativa de Clustering (Hito 1 vs Hito 2)

El clustering se evaluó mediante el coeficiente Silhouette:

| Experimento | Características de Entrada | Coeficiente Silhouette | Interpretación / Significado |
| :--- | :--- | :---: | :--- |
| **Hito 1** | Features Manuales Geométricas | **0.752** | **Espejismo Metodológico**: Agrupa lienzos basados en dimensiones físicas (retrato vs. paisaje). Alta separación matemática, pero nula relevancia artística. |
| **Hito 2** | Embeddings Visuales ResNet-18 | **0.079** | **Resultado Real**: Agrupa según información semántica e visual real. El score bajo refleja que los estilos pictóricos forman un continuo visual superpuesto. |

> [!NOTE]
> Aunque numéricamente 0.079 es inferior a 0.752, representa un **salto metodológico correcto**: los movimientos artísticos (ej. Impresionismo y Postimpresionismo) no se dividen en clusters matemáticamente densos y aislados, sino que comparten patrones, paletas y técnicas comunes. El sweep de K muestra que el Silhouette score es bajo pero estable en el rango 0.07–0.09 para K=2..5, validando que el espacio visual continuo de arte tiene alta complejidad.

---

## 🔍 6. Interpretabilidad Visual con Grad-CAM

* **Capa objetivo:** `resnet.layer4[-1]` (último bloque convolucional).
* **Resultado:** Se ha generado el mapa de calor exitosamente sobre una obra real del DataLoader sin ninguna advertencia en consolas gracias a la corrección de hooks (`register_full_backward_hook`).

> [!CAUTION]
> **Advertencia de Interpretabilidad (Modelo sin Fine-Tuning):**
> Es importante destacar que estamos utilizando una red ResNet-18 preentrenada en ImageNet (un dataset de fotografía general) **sin realizar fine-tuning sobre WikiArt**. Por lo tanto, los mapas de calor generados por Grad-CAM muestran qué características visuales generales (bordes, contrastes, formas básicas) capturan la atención de un extractor de características de propósito general.
> 
> No deben interpretarse como "trazos o estilos artísticos aprendidos", sino como el comportamiento del modelo baseline preentrenado. Esto sienta las bases para el Hito 3, donde el fine-tuning de la red sobre arte permitirá que Grad-CAM resalte elementos estilísticos específicos (como la textura de pinceladas o parches de color).

---

## 🖼️ 7. Q1: Clasificación de Estilo usando Embeddings de ResNet-18

Para responder a la pregunta de clasificación del estilo artístico (Q1) utilizando deep learning, se entrenaron clasificadores supervisados sobre la muestra de 10,000 embeddings extraídos (split 80/20 train/test).

### Comparación de Clasificación (Hito 1 vs. Hito 2)

| Clasificador | Características de Entrada | F1-Score Macro | Accuracy | Observación |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest Baseline** | Features Geométricas (Tabulares) | **0.138** | 0.24 | Colapso a clases mayoritarias por falta de señal. |
| **Random Forest Hito 1** | Features Visuales Manuales | **0.236** | — | Muestra de 3k. Features de color/bordes tradicionales. |
| **Random Forest Hito 2** | Embeddings ResNet-18 (512-d) | **0.330** | 0.40 | Reducción de varianza al escalar la muestra a 10k. |
| **Regresión Logística Hito 2** | Embeddings ResNet-18 (512-d) | **0.430** | 0.45 | **Mejor modelo**. Excelente rendimiento al normalizar características con `StandardScaler`. |

> [!TIP]
> **Conclusión Clave:** Al pasar de características geométricas a descriptores profundos en una muestra de escala real (10,000 imágenes), el F1-Score macro subió de **0.138 a 0.430 (+29.2% absoluto)** y superó por **+19.4%** al mejor clasificador del Hito 1 con características visuales manuales. Esto demuestra de forma cuantitativa el enorme poder de representación de las características semánticas profundas para discriminar estilos artísticos.

---

## 📅 8. Q3: Regresión del Año de Creación usando Embeddings

Se recuperó el análisis de regresión del año de creación de la obra (Q3), utilizando la columna `description` para extraer el año de creación (5,844 obras con año válido en la muestra de 10,000). Se entrenaron regresores sobre los embeddings de 512 dimensiones (split 80/20).

### Comparación Cuantitativa de MAE (Error Absoluto Medio)

| Regresor | Características de Entrada | MAE (Error Medio) | Reducción de Error |
| :--- | :--- | :---: | :---: |
| **Random Forest Baseline (Hito 1)** | Features Geométricas (Tabulares) | **81.40 años** | Baseline |
| **Ridge Regressor (Hito 2)** | Embeddings ResNet-18 (512-d) | **59.46 años** | -21.94 años |
| **Random Forest Regressor (Hito 2)** | Embeddings ResNet-18 (512-d) | **53.57 años** | **-27.83 años** |

> [!TIP]
> **Conclusión Clave:** Utilizar embeddings visuales de aprendizaje profundo en una muestra grande (10k) redujo el error absoluto medio (MAE) de predicción de año de **81.40 años a solo 53.57 años** (un recorte masivo de más de 27 años de error). Esto evidencia que la evolución temporal de los estilos artísticos (trazos, paletas de colores, composición histórica) queda codificada de forma cuantitativa en el espacio latente de la ResNet-18.

---

## 🚀 9. Plan de Trabajo Futuro: Hito 2 a Hito 3

Con la validación cuantitativa completa, las líneas de desarrollo para el Hito 3 quedan firmemente estructuradas:

1. **Fine-Tuning de ResNet-18:**
   Entrenar la red convolucional de extremo a extremo directamente sobre WikiArt para clasificar géneros artísticos. Esto adaptará los filtros convolucionales a pinceladas y texturas artísticas.
2. **Impacto en Grad-CAM:**
   El fine-tuning permitirá que los heatmaps resalten los trazos, espátulas e iluminaciones específicos que definen cada estilo (ej. pinceladas cortas impresionistas) en lugar de contornos de objetos generales.
3. **Optimización Multietiqueta con Embeddings:**
   Extender el clasificador multilabel utilizando los embeddings visuales combinados con la calibración del umbral probabilístico por clase, superando el colapso del baseline geométrico.
