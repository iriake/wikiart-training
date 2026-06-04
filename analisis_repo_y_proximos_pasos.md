# Análisis del Repositorio WikiArt — Estado Actual y Próximos Pasos

**Proyecto:** CC5205 Minería de Datos — Grupo 6, Sección 3  
**Fecha:** 2026-06-04

---

## 📁 Estructura del Repositorio

| Archivo / Directorio | Descripción |
|---|---|
| [wikiart_hito1.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito1.ipynb) | Notebook del Hito 1 (EDA + modelos baseline) |
| [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb) | Notebook del Hito 2 (mejoras, desbalance, multilabel, ResNet, clustering, Grad-CAM) |
| [CSVs/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/CSVs) | `classes.csv`, `visual_features.csv`, `wclasses.csv` |
| [reports/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/reports) | Informes HTML, instrucciones del hito 2, presentación PDF |
| [outputs/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/outputs) | `artists_count.txt`, `genres_count.txt` |
| [src/](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/src) | Solo `__init__.py` (módulo vacío) |
| `requirements.txt` | Dependencias del proyecto |

---

## ✅ Estado del Hito 2 — Notebook Ejecutado

El notebook [wikiart_hito2.ipynb](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/wikiart_hito2.ipynb) **fue ejecutado completamente** (todas las celdas tienen `execution_count` asignados, del 1 al 13, con outputs generados). A continuación el análisis por sección:

### 1. Análisis de la Matriz de Confusión (Hito 1)

- **Datos usados:** Top 10 géneros principales del CSV de metadatos (~80k pinturas).
- **Modelo baseline:** RandomForest (100 estimadores, max_depth=12) sobre features tabulares geométricas (aspect_ratio, log_width, log_height, is_portrait, is_square, es_multilabel).
- **Resultado baseline:**
  - Accuracy: **24%**
  - Macro avg F1: **0.13**
  - El modelo colapsa hacia **Impressionism** (recall 0.66) y **Realism** (recall 0.42), mientras que clases minoritarias como **Abstract Expressionism** (recall 0.04) y **Northern Renaissance** (recall 0.02) son prácticamente ignoradas.

> [!IMPORTANT]
> La conclusión es clara: las features tabulares geométricas son insuficientes para distinguir géneros artísticos. El modelo simplemente predice la clase mayoritaria.

### 2. Estrategias de Desbalance de Clases

Se implementaron dos técnicas:

1. **Class Weighting** (`class_weight='balanced'`):
   - Accuracy baja a **15%**, pero el macro avg recall sube a **0.18** (vs 0.15 baseline).
   - Genera predicciones más distribuidas entre clases, pero pierde precisión global.

2. **Random Undersampling** (recortando clases mayoritarias a la mediana):
   - Accuracy: **23%** (similar al baseline).
   - Macro avg F1 sube a **0.20** — mejor balance entre clases.
   - Es la estrategia más equilibrada de las dos.

> [!NOTE]
> Ambas técnicas confirman que el problema no es solo el desbalance sino la **pobreza de las features tabulares** para esta tarea.

### 3. Clasificación Multietiqueta (Multilabel)

- Se utilizó `MultiLabelBinarizer` sobre el campo `genre_list` (27 clases).
- Modelo: `OneVsRestClassifier(RandomForestClassifier(n_estimators=50, max_depth=10))`.
- **Resultado devastador:**
  - Micro avg F1: **0.01**
  - La mayoría de clases tienen recall **0.00**.
  - Solo `Color Field Painting` (precision 0.69, recall 0.03) y `Minimalism` (precision 0.50, recall 0.01) muestran alguna señal.
  
- Se generaron matrices de confusión binarias por clase para 6 géneros representativos.

> [!WARNING]
> El enfoque multilabel con features tabulares es **completamente inviable**. Las features geométricas no capturan información visual suficiente para resolver este problema.

### 4. Extracción de Embeddings con ResNet-18

- Se usó ResNet-18 preentrenada (ImageNet) para extraer embeddings de 512 dimensiones.
- Se simuló la extracción con datos aleatorios (`torch.randn`) ya que no se tenían las imágenes disponibles localmente.
- Se entrenó un RandomForest sobre los embeddings simulados.
- **Resultado (con datos simulados):**
  - Accuracy: **10%** — como es de esperar con features aleatorias.

> [!CAUTION]
> **Los embeddings fueron generados con datos aleatorios, NO con imágenes reales.** Los resultados de esta sección y las posteriores (clustering, Grad-CAM) **no son válidos** para conclusiones reales del proyecto.

### 5. Clustering con Embeddings ResNet

- Se aplicó KMeans (k=10) sobre los embeddings simulados de ResNet-18.
- Coeficiente Silhouette: **0.013** — extremadamente bajo, lo cual es esperado con datos aleatorios.
- Se generó una visualización PCA 2D del clustering.

### 6. Interpretabilidad con Grad-CAM

- Se implementó una clase `GradCAM` que registra gradientes y activaciones de `layer4` de ResNet-18.
- La clase está correctamente definida e instanciada, pero **no se generaron visualizaciones** reales porque no había imágenes reales para procesar.

---

## 📊 Resumen de Resultados Obtenidos

| Experimento | Accuracy | Macro F1 | Observación |
|---|---|---|---|
| Baseline tabular (RF) | 24% | 0.13 | Colapso a clases mayoritarias |
| Class weighting (RF) | 15% | 0.15 | Mejor recall pero peor accuracy |
| Undersampling (RF) | 23% | 0.20 | Mejor balance general |
| Multilabel (OvR RF) | — | 0.01 | Completamente inviable con features tabulares |
| ResNet-18 embeddings | 10% | — | ⚠️ Datos simulados, no válidos |
| Clustering embeddings | Silhouette 0.013 | — | ⚠️ Datos simulados, no válidos |

---

## 🚀 Próximos Pasos Recomendados

### Prioridad Alta

1. **Obtener y procesar las imágenes reales**
   - Descargar las imágenes del dataset WikiArt (al menos las del subconjunto de los Top 10 géneros, ~60k imágenes).
   - Verificar que las rutas en el CSV (`classes.csv`) coincidan con las imágenes descargadas.
   - Sin imágenes reales, las secciones 4, 5 y 6 del Hito 2 **carecen de validez**.

2. **Re-ejecutar la extracción de embeddings con imágenes reales**
   - Reemplazar `torch.randn` por el pipeline real de carga y transformación de imágenes.
   - Guardar los embeddings extraídos en un CSV o archivo `.npy` para reutilizarlos sin re-computar.

3. **Re-evaluar clasificación y clustering con embeddings reales**
   - Entrenar RandomForest (y potencialmente otros modelos como SVM, XGBoost) sobre embeddings reales.
   - Re-calcular métricas de clustering (Silhouette, ARI vs etiquetas reales).

### Prioridad Media

4. **Generar visualizaciones Grad-CAM reales**
   - Seleccionar muestras representativas de cada género.
   - Generar heatmaps que muestren qué regiones de las pinturas influyen en la clasificación.
   - Esto es crucial para la **interpretabilidad cualitativa** requerida en las instrucciones del Hito 2.

5. **Mejorar el análisis cualitativo**
   - Mostrar ejemplos concretos donde el modelo acierta y falla.
   - Asignar nombres descriptivos a los clusters.
   - Comparar clustering sobre features manuales (Hito 1) vs embeddings profundos (Hito 2).

6. **Considerar fine-tuning de ResNet-18**
   - En lugar de solo extraer embeddings, hacer fine-tuning de las últimas capas para la tarea específica de clasificación de géneros artísticos.
   - Esto podría mejorar significativamente los resultados.

### Prioridad Baja

7. **Ampliar el análisis multilabel con embeddings profundos**
   - Repetir el experimento multilabel usando embeddings de ResNet en lugar de features tabulares.

8. **Preparar la presentación final**
   - Según las [instrucciones del Hito 2](file:///C:/Users/ko5/Documents/Proyectos/wikiart-training/reports/instrucciones_hito2.md):
     - Motivación y objetivos
     - Preguntas abordadas
     - Experimentos y resultados (cuantitativos + cualitativos)
     - Futuras direcciones
   - Reporte máximo 15 páginas + anexos ilimitados.

9. **Documentar contribuciones individuales del grupo**
   - Requerido explícitamente en las instrucciones.

---

## ⚠️ Problema Crítico

> [!CAUTION]
> El obstáculo principal del proyecto es que **las secciones 4, 5 y 6 del notebook del Hito 2 fueron ejecutadas con datos simulados (aleatorios)**, no con imágenes reales. Antes de la entrega final, es **imprescindible**:
> 1. Obtener acceso a las imágenes del dataset.
> 2. Re-ejecutar todo el pipeline de embeddings con datos reales.
> 3. Actualizar todas las métricas y visualizaciones correspondientes.
>
> Sin esto, las conclusiones sobre deep learning e interpretabilidad **no tienen validez científica**.
