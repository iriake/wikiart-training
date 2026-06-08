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

> [!CAUTION]
> **Nota de la versión anterior (Simulada):**
> En ejecuciones previas del notebook, el Silhouette score reportaba **0.965**. Esto era un **artefacto numérico falso** causado porque el directorio `./images/` estaba vacío y el dataset sustituía silenciosamente los archivos faltantes por imágenes negras constantes. Esto forzaba a la ResNet a generar el mismo vector de embeddings y agrupaba ruido infinitesimal.

### Estado Actual:
* Las imágenes ya han sido **totalmente descargadas** en `./images/`.
* La infraestructura está lista para extraer embeddings visuales reales sobre GPU usando el notebook [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb). Se espera que la métrica de Silhouette real se ubique en el rango típico de 0.1–0.4 una vez ejecutado en un hardware adecuado.

---

## 🔍 6. Interpretabilidad Visual con Grad-CAM

* **Capa objetivo:** `resnet.layer4[-1]` (último bloque convolucional).
* **Estado:** Se ha corregido la dependencia faltante instalando `opencv-python` (OpenCV) y registrando los hooks correspondientes. El pipeline generará mapas de calor reales una vez que el notebook se ejecute con las imágenes cargadas.

---

## 🚀 7. Estado del Proyecto y Próximos Pasos

> [!NOTE]
> **Resumen de Estado:** El notebook unificado de Hito 2 está completamente corregido, las dependencias de OpenCV están resueltas en el entorno y las imágenes han sido totalmente descargadas. La limitación actual es que la máquina local no cuenta con suficiente potencia gráfica para completar el proceso de inferencia de ResNet-18 y el cálculo de matrices de confusión.

### Acciones Completadas
1. **Unificación de Notebooks**: Integración de todo el flujo y la técnica 3 de balanceo en [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb).
2. **Descarga de Datos**: 81,444 imágenes reales preparadas localmente.
3. **Instalación de Dependencias**: Agregado `opencv-python` en [requirements.txt](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/requirements.txt) e instalado en el sistema.

### Próximos Pasos (Para ejecutar en la máquina del compañero)
1. **Ejecución Completa**: Correr todas las celdas del notebook para actualizar los outputs y gráficos con imágenes reales.
2. **Registro de Silhouette Real**: Anotar el coeficiente de Silhouette real del clustering sobre embeddings de imágenes reales.
3. **Análisis Cualitativo**: Inspeccionar y exportar los heatmaps de Grad-CAM generados sobre obras reales para identificar qué patrones (formas, colores) guían la clasificación de estilos.
4. **Preparación del Reporte**: Redactar el informe escrito (máximo 15 páginas) y preparar la presentación de diapositivas basándose en los resultados reales obtenidos.
