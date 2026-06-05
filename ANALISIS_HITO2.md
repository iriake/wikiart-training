# Análisis de Resultados — WikiArt Hito 2

> Análisis de los resultados obtenidos al ejecutar [wikiart_hito2_claude.ipynb](wikiart_hito2_claude.ipynb).
> Dataset: 76.5k pinturas etiquetadas por género, restringido a las 10 clases más frecuentes
> (61.208 obras para clasificación, 12.242 en test).

---

## 1. Análisis del Baseline (Hito 1) — Random Forest sobre features tabulares

### Distribución del Top-10 de géneros

| Rank | Género | # Pinturas |
|------|--------------------------|-----------:|
| 1 | Impressionism | 13.028 |
| 2 | Realism | 10.546 |
| 3 | Romanticism | 6.919 |
| 4 | Expressionism | 6.335 |
| 5 | Post Impressionism | 6.307 |
| 6 | Symbolism | 4.524 |
| 7 | Baroque | 4.236 |
| 8 | Art Nouveau Modern | 4.168 |
| 9 | Abstract Expressionism | 2.594 |
| 10 | Northern Renaissance | 2.551 |

El **ratio de desbalance es 5.1×** entre la clase mayoritaria (Impressionism) y la minoritaria
(Northern Renaissance). No es un desbalance extremo en términos absolutos, pero sí suficiente
para que un modelo sin contramedidas colapse hacia las clases dominantes.

### Resultados del baseline

| Métrica global | Valor |
|----------------|------:|
| Accuracy | **0.24** |
| F1 macro | **0.13** |
| F1 weighted | 0.18 |

### Análisis por clase (baseline)

| Género | Precision | Recall | F1 | Observación |
|---|---:|---:|---:|---|
| Impressionism | 0.27 | **0.66** | 0.38 | Atrae 2/3 de las predicciones reales |
| Realism | 0.23 | **0.42** | 0.29 | Segundo "sumidero" |
| Symbolism | 0.38 | 0.07 | 0.11 | Alta precisión, pero casi no predicha |
| Post Impressionism | 0.17 | **0.02** | 0.03 | Prácticamente ignorada |
| Northern Renaissance | 0.20 | **0.02** | 0.04 | Ignorada (minoritaria) |
| Abstract Expressionism | 0.22 | **0.04** | 0.07 | Ignorada (minoritaria) |

**Diagnóstico:** las features tabulares son puramente **geométricas**
(`aspect_ratio`, `log_width`, `log_height`, `is_portrait`, `is_square`, `es_multilabel`).
Estas variables no contienen información discriminativa entre estilos artísticos: todos los
movimientos pictóricos usan formatos físicos similares. El modelo, ante la ausencia de
señal, recurre a la estrategia trivial de **predecir las clases mayoritarias**.
Esto se ve claramente en la matriz de confusión: las columnas de Impressionism y Realism
acumulan la mayor parte de la masa de predicciones, sin importar la clase real.

---

## 2. Estrategias de Balanceo de Clases

Se compararon tres técnicas sobre el mismo `X_te` desbalanceado para una evaluación justa.

### 2.1 Resumen comparativo (métricas globales)

| Modelo | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| **Baseline (sin balancear)** | **0.24** | 0.13 | 0.18 |
| Class Weighting (`class_weight='balanced'`) | 0.15 | 0.15 | 0.15 |
| **Undersampling manual** (a la mediana) | 0.23 | **0.20** | **0.22** |
| BalancedRandomForestClassifier (imblearn) | 0.15 | 0.15 | 0.15 |

### 2.2 Hallazgos clave

**A. El undersampling manual fue la mejor técnica de balanceo.**
Es el único método que **mejora simultáneamente F1-macro y F1-weighted** respecto al
baseline, sin sacrificar accuracy de forma drástica (0.24 → 0.23). Esto se debe a que
recorta cada clase mayoritaria hasta la mediana (4.524 muestras) en lugar de eliminar el
máximo posible, manteniendo un dataset todavía grande (45.148 muestras).

**B. `class_weight='balanced'` y `BalancedRandomForestClassifier` produjeron resultados
casi indistinguibles** (F1-macro 0.15 / accuracy 0.15 en ambos). Comparando clase a clase:

| Género | Class Weight (R / F1) | Balanced RF (R / F1) |
|---|---:|---:|
| Abstract Expressionism | 0.20 / 0.13 | 0.20 / 0.12 |
| Art Nouveau Modern | 0.20 / 0.17 | 0.20 / 0.17 |
| Baroque | 0.19 / 0.14 | 0.22 / 0.14 |
| Impressionism | 0.12 / 0.17 | 0.12 / 0.17 |
| Northern Renaissance | 0.33 / 0.13 | 0.31 / 0.13 |
| Realism | 0.10 / 0.15 | 0.10 / 0.14 |

Esta convergencia tiene una explicación teórica: ambas técnicas, en última instancia,
reescalan la influencia de cada clase en el aprendizaje, ya sea **a nivel de impureza
(class_weight)** o **a nivel de bootstrap (BalancedRF)**. Con un dataset donde la señal
visual no existe, el efecto neto es el mismo: el modelo se vuelve "valiente" prediciendo
clases minoritarias, sube su recall, pero pierde toda la precisión que tenía al
predecir las mayoritarias.

**C. El trade-off accuracy ↔ recall minoritario es muy marcado.**
Por ejemplo, en Northern Renaissance:
- Baseline: recall 0.02, F1 0.04
- Class Weight: recall **0.33**, F1 0.13 (×3 más F1, ×16 más recall)
- BRF: recall **0.31**, F1 0.13

Pero a cambio, Impressionism pierde la mayoría de sus aciertos:
- Baseline: recall 0.66, F1 0.38
- Métodos balanceados: recall 0.12, F1 0.17

### 2.3 Conclusión de la sección

El experimento confirma que **el balanceo de clases no compensa la falta de features
discriminativas**. Las tres técnicas sólo redistribuyen errores: empujan al modelo a
predecir clases minoritarias, pero no le dan la información que le falta para acertar.
El techo de 0.20–0.24 en F1-macro / accuracy refleja el límite real de lo que se puede
aprender con 6 variables geométricas en este problema. Esto **motiva fuertemente la
sección 4 (embeddings de ResNet)**, donde recién se introducen descriptores visuales
genuinos.

---

## 3. Clasificación Multietiqueta (Multilabel)

| Configuración | Valor |
|---|---|
| Estrategia | OneVsRest + Random Forest (50 árboles, profundidad 10) |
| # clases binarias | 27 |
| Tamaño train | 64.033 × 6 features |

**Resultados:**

| Métrica | Valor |
|---|---:|
| Micro avg F1 | 0.01 |
| Macro avg F1 | 0.01 |
| Weighted avg F1 | 0.01 |
| Samples avg F1 | 0.00 |

**El modelo prácticamente no emite predicciones positivas en ninguna clase.**
Sólo las cabezas binarias de las clases con cierta señal alcanzan recalls de 0.02–0.03
(Symbolism, Color Field Painting). Las advertencias de `UndefinedMetricWarning`
confirman que **hay clases donde no se predice ni una sola muestra positiva** en todo el
test.

**Causa raíz:** combinar
1. Features geométricas sin poder discriminativo,
2. Umbral de decisión por defecto en 0.5,
3. Clases con prevalencia bajísima (Action painting tiene 20 muestras en test, Synthetic
   Cubism 51),

resulta en que para casi ninguna pintura la probabilidad estimada de pertenecer a un género
supera 0.5. El clasificador, en términos prácticos, devuelve siempre "no pertenece a
ninguna clase".

**Lo que se podría hacer en el Hito 3:**
- Calibrar umbrales por clase (p. ej. al cuantil que iguale recall y precision).
- Usar las probabilidades en lugar de las predicciones binarias (`predict_proba`) y
  evaluar con `average_precision_score`.
- Repetir el ejercicio con los embeddings de ResNet en lugar de las features tabulares.

---

## 4. Embeddings de ResNet-18 (Deep Learning)

| Parámetro | Valor |
|---|---|
| Modelo | ResNet-18 pre-entrenada en ImageNet, sin fine-tuning |
| Dimensión del embedding | 512 |
| Dispositivo | CUDA (GPU) |
| Tamaño de muestra | 2.974 |
| Tiempo de extracción | ~9.5 s (≈ 310 img/s en GPU) |

> ⚠️ **Las imágenes del dataset aún no están descargadas en `./images/`.**
> El pipeline ejecuta sin lanzar errores porque `WikiArtImageDataset.__getitem__` tiene
> un `try/except` que sustituye cada archivo faltante por una imagen negra de 224×224:
>
> ```python
> try:
>     image = Image.open(img_path).convert('RGB')
> except Exception:
>     image = Image.new('RGB', (224, 224))   # ← placeholder negro
> ```
>
> Por lo tanto, **los 2.974 embeddings extraídos son todos prácticamente idénticos** (la
> ResNet-18 aplicada a una imagen negra normalizada con stats de ImageNet produce siempre
> el mismo vector de 512 dimensiones). Esto **invalida los resultados numéricos de las
> secciones 5 y 6**: lo único que valida esta corrida es que la **infraestructura**
> (modelo en GPU, DataLoader, transformaciones, hook de Grad-CAM) está bien armada.

**Lo que esta sección sí confirma:**
- ResNet-18 se descarga y carga correctamente sobre CUDA.
- El pipeline `Dataset → DataLoader → modelo → CPU → numpy` funciona end-to-end.
- El throughput de extracción (~310 img/s) deja claro que escalar al dataset completo
  (80k imágenes ≈ 4.3 min en GPU) es perfectamente viable una vez se tengan los archivos.

---

## 5. Comparación de Clustering (Features manuales vs Embeddings ResNet)

| Métrica | Valor |
|---|---:|
| K (clusters) | 4 |
| Reducción previa | PCA a 50 componentes |
| Silhouette score | 0.965 |

### ⚠️ Resultado no interpretable

El silhouette de **0.965 es un artefacto, no un hallazgo.** Como se explicó en la sección
anterior, todos los embeddings de entrada son esencialmente el mismo vector (imagen negra
placeholder). En ese escenario, KMeans encuentra "clusters" sobre ruido numérico
infinitesimal: la métrica se dispara porque todos los puntos de cada cluster son casi
idénticos entre sí (cohesión perfecta) y los pocos outliers reales (si los hubiera por
diferencias en `convert('RGB')` o variaciones de codificación) caen a distancias
relativamente enormes.

**Esta sección no puede compararse honestamente con el clustering del Hito 1** hasta que
las imágenes estén disponibles. Lo que sí se puede reportar es:

- El código de extracción → PCA → KMeans → silhouette está correctamente conectado.
- Una vez se descarguen las imágenes, basta con re-ejecutar la celda 26 sin modificarla
  para obtener un silhouette real (esperable en el rango 0.1–0.4, típico de imágenes
  artísticas en espacio ResNet).

**Recomendación adicional (independiente de tener las imágenes):** agregar al `__getitem__`
un contador de fallos o, mejor, **lanzar una excepción explícita** si la imagen no existe,
en lugar de devolver silenciosamente un placeholder. El `try/except` actual oculta el
problema y produce resultados falsamente exitosos como este 0.965.

---

## 6. Interpretabilidad con Grad-CAM

| Etapa | Estado |
|---|---|
| Instanciación de `GradCAM` sobre `resnet.layer4[-1]` | ✓ OK |
| Generación de heatmap sobre imagen real | ✗ No evaluable |

Hay dos motivos por los que esta sección no produce un resultado útil en esta corrida:

1. **`ModuleNotFoundError: No module named 'cv2'`** — falta instalar OpenCV:
   ```bash
   pip install opencv-python
   ```
2. **Aunque cv2 estuviera instalado**, sin las imágenes descargadas el Grad-CAM se
   generaría sobre la imagen negra placeholder y el heatmap resultante no tendría
   significado interpretativo alguno (ResNet activaría regiones arbitrarias sobre una
   entrada constante).

Para que esta sección entregue valor, se necesitan **las dos cosas**: instalar cv2 **y**
tener las imágenes locales. Lo que sí se valida ahora es que los hooks de forward y
backward sobre `layer4[-1]` quedan correctamente registrados.

---

## Conclusiones generales

1. **Las features geométricas tienen un techo de rendimiento de ~0.24 accuracy / 0.20 F1-macro**.
   Ninguna técnica de balanceo lo cambia sustancialmente; sólo redistribuyen errores entre
   clases.

2. **El undersampling manual fue el mejor método de balanceo en este experimento**,
   superando tanto al class-weight como al BalancedRandomForestClassifier. La razón
   probable es que recorta a la mediana (preservando 73% de los datos) en lugar de
   balancear a la clase más pequeña.

3. **Class-weight y Balanced Random Forest producen resultados prácticamente
   indistinguibles** en este dataset, evidencia empírica de que ambos mecanismos
   (reescalado de impureza vs. submuestreo intra-bagging) son funcionalmente equivalentes
   cuando los descriptores carecen de poder discriminativo.

4. **La multilabel con features tabulares no funciona** (F1 ≈ 0.01). Es un caso de libro
   de cómo el umbral por defecto de 0.5 colapsa OneVsRest sobre clases poco prevalentes.

5. **El pipeline de Deep Learning (ResNet-18 + Grad-CAM) está correctamente armado** y
   es escalable (~310 img/s en GPU → ~4.3 min para el dataset completo de 80k), pero
   **en esta corrida no produce resultados interpretables** porque el directorio
   `./images/` está vacío. El `try/except` del Dataset oculta el problema sustituyendo
   las imágenes faltantes por placeholders negros, lo que invalida tanto el clustering
   (silhouette 0.965 = artefacto) como cualquier visualización Grad-CAM. Lo único que se
   valida hoy de las secciones 4–6 es la infraestructura (carga del modelo, DataLoader,
   hooks de gradiente).

### Próximos pasos para el Hito 3

**Bloqueantes (sin los cuales nada de la parte visual cuenta):**
- **Descargar el dataset WikiArt** en `./images/`.
- **Endurecer el `__getitem__`**: que falle explícitamente si una imagen no existe, en
  lugar de devolver placeholders. Esto previene resultados falsamente exitosos como el
  silhouette de 0.965.
- **Instalar `opencv-python`** para completar Grad-CAM (o reemplazar cv2 por
  `scipy.ndimage.zoom` + `matplotlib.cm.jet`).

**Una vez resueltos los bloqueantes:**
- Re-entrenar los clasificadores de la sección 2 usando los embeddings de ResNet como
  features. Ahí debería verse el salto cualitativo de accuracy (el techo de 0.24 viene
  de las features geométricas, no del modelo).
- Volver a evaluar el balanceo con esos embeddings. Muy probablemente el
  `BalancedRandomForestClassifier` dejará de ser equivalente al `class_weight` cuando
  exista señal visual real que distinga las clases.
- Re-correr el clustering y comparar el silhouette real contra el del Hito 1.
- Calibrar umbrales por clase en la sección multilabel (`predict_proba` + cuantiles en
  lugar del 0.5 por defecto).
