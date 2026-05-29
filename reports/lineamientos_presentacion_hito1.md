# Lineamientos de Exposición — WikiArt Hito 1
**Proyecto de Minería de Datos (CC5205) — Grupo 6, Sección 3**

Este documento contiene la estructura sugerida, el guión diapositiva por diapositiva y una guía de defensa frente a preguntas de la comisión para la presentación del **Hito 1: Análisis de Pinturas WikiArt con Machine Learning**.

---

## 📌 Recomendaciones Generales para la Presentación
* **Tiempo Objetivo:** 5 a 6 minutos (aprox. 1 minuto por diapositiva principal).
* **Distribución de Roles:** Se sugiere que 2 o 3 integrantes expongan la parte principal (Diapositivas 1 a 6) para mantener el ritmo, mientras que los demás apoyan en la sección de preguntas utilizando los Anexos (Diapositivas 8 a 16).
* **Tono Académico y Técnico:** Usen terminología precisa (ej. *desbalance de clases*, *F1-score macro*, *baseline*, *Silhouette score*, *features tabulares/visuales*, *embeddings*).
* **Control del Ritmo:** No lean las diapositivas. Úsenlas como apoyo visual mientras explican la lógica e interpretación de los datos y modelos.

---

## 🧭 Guión Diapositiva por Diapositiva

### 🛈 Diapositiva 1: Portada (Título y Equipo)
* **Tiempo Sugerido:** 20 - 30 segundos.
* **Apoyo Visual:** Título "WikiArt", número de grupo, sección y lista de integrantes.
* **Guión del Orador:**
  > *"Buenos días profesor y compañeros. Nosotros somos el Grupo 6 de la Sección 3, integrado por Felipe Ramírez, Iñaki Ramírez, José Luis Espinoza, Macarena Guerra y Juan Pablo Carrasco. Hoy les presentaremos los avances del Hito 1 de nuestro proyecto: 'WikiArt — Análisis de Pinturas con Machine Learning'. En esta primera etapa, nos enfocamos en el análisis exploratorio del conjunto de datos, la ingeniería de características visuales y tabulares básicas, y el planteamiento experimental para abordar cuatro preguntas clave sobre clasificación, clustering, regresión temporal e interpretabilidad."*

---

### 🖼 Diapositiva 2: Introducción al Proyecto
* **Tiempo Sugerido:** 45 - 60 segundos.
* **Apoyo Visual:** Definición de WikiArt como enciclopedia visual, mención de su valor como repositorio colaborativo y mención a estilos clásicos (Impresionismo, Realismo, Romanticismo).
* **Guión del Orador:**
  > *"Para contextualizar, WikiArt es una de las mayores enciclopedias visuales de arte en línea, albergando miles de pinturas y esculturas de museos y colecciones privadas. Clasificar y analizar estas obras de forma manual requiere de expertos y mucho tiempo. 
  >
  > Nuestro proyecto aborda este problema desde la minería de datos: ¿es posible utilizar machine learning para identificar patrones estéticos y temporales que correspondan a movimientos artísticos reconocidos? Con esto buscamos automatizar la catalogación en museos, descubrir nuevas taxonomías visuales que la historia del arte tradicional no ha catalogado y entender qué características visuales definen a cada estilo o época."*

---

### 📊 Diapositiva 3: Exploración del Dataset
* **Tiempo Sugerido:** 60 - 75 segundos.
* **Apoyo Visual:** Cifras clave (80,042 obras, 1,119 artistas, 27 géneros artísticos, rango temporal de 1204 a 2012). Mención sobre la extracción de características cromáticas, de brillo y contraste desde los píxeles.
* **Guión del Orador:**
  > *"Trabajamos con un dataset consolidado de más de 80 mil pinturas. La distribución temporal abarca desde el año 1204 al 2012, aunque hay un fuerte sesgo y concentración hacia finales del siglo XIX e inicios del XX. 
  > 
  > El dataset presenta un marcado desbalance de clases, donde los géneros dominantes son el Impresionismo y el Realismo. Para complementar la información estructural del CSV (como las dimensiones físicas), implementamos una extracción de variables visuales manuales desde los píxeles en una muestra representativa de 3,000 imágenes, calculando métricas de color en espacios RGB y HSV, así como contraste, brillo y calidez tonal, lo que nos da un punto de partida interpretable antes de pasar a modelos de Deep Learning."*

---

### ❓ Diapositiva 4: Preguntas de Investigación
* **Tiempo Sugerido:** 40 - 45 segundos.
* **Apoyo Visual:** Las cuatro preguntas estructuradas (P1 a P4).
* **Guión del Orador:**
  > *"Para estructurar nuestro análisis, definimos cuatro preguntas fundamentales:
  > * **P1 (Clasificación):** ¿Qué tan bien podemos clasificar el estilo artístico de una pintura usando sus metadatos tabulares y características visuales básicas?
  > * **P2 (Clustering):** ¿Existen grupos de pinturas con características visuales similares pero que pertenecen a distintos estilos históricos?
  > * **P3 (Regresión):** ¿Es posible estimar el año de creación de una obra a partir de su geometría o sus colores?
  > * **P4 (Interpretabilidad):** ¿Qué características físicas o visuales caracterizan y diferencian numéricamente a cada estilo artístico?"*

---

### ⚙ Diapositiva 5: Propuesta Experimental
* **Tiempo Sugerido:** 60 - 75 segundos.
* **Apoyo Visual:** Relación de tareas y modelos (Clasificación con Árboles de Decisión y Random Forest; Clustering con K-Means y Silhouette; Regresión con Random Forest Regressor; Interpretabilidad mediante Feature Importance).
* **Guión del Orador:**
  > *"Nuestra propuesta metodológica aborda cada pregunta con un pipeline de minería de datos específico:
  > * Para la **Clasificación (P1)**, implementamos Árboles de Decisión y Random Forest sobre los 10 géneros más representados para mitigar el desbalance, comparando un baseline tabular versus un modelo enriquecido con las variables visuales.
  > * En **Clustering (P2)**, aplicamos K-Means sobre las variables tabulares estandarizadas de todo el dataset, utilizando el análisis de Silhouette y el método del codo para determinar el número de agrupaciones naturales.
  > * Para la **Regresión (P3)**, entrenamos un Random Forest Regressor sobre las obras que cuentan con año de creación válido para estimar el año cronológico.
  > * Finalmente, para la **Interpretabilidad (P4)**, analizamos la importancia de características (Feature Importance) de los clasificadores y cruzamos las medias geométricas por género artístico para validar coherencias históricas."*

---

### 🏆 Diapositiva 6: Resultados Preliminares
* **Tiempo Sugerido:** 100 - 120 segundos.
* **Apoyo Visual:** Métricas preliminares. Clasificación (DT F1=0.124, RF F1=0.138, RF+visuales F1=0.236). Clustering (K=4 óptimo, Silhouette=0.752). Regresión (MAE = 67.7 años vs baseline de 81.6 años). Interpretabilidad (aspect_ratio e is_portrait son las más relevantes).
* **Guión del Orador:**
  > *"Los resultados de este primer hito nos entregan revelaciones críticas:
  >
  > En **Clasificación (P1)**, los atributos tabulares geométricos solos rinden muy bajo (F1-score de 0.138). Sin embargo, al incorporar las características visuales (canales RGB, saturación y brillo), el F1-score sube notablemente a **0.236** en la muestra, demostrando que el color es mucho más informativo para los estilos que las dimensiones del lienzo.
  >
  > Para el **Clustering (P2)**, K-Means identificó **K = 4** como óptimo con un alto coeficiente Silhouette de **0.752**. El hallazgo clave aquí es que estos clusters representan agrupamientos geométricos sumamente definidos (ej. formatos verticales versus horizontales masivos), pero al cruzarlos con la historia del arte, los estilos se mezclan de forma homogénea. Esto sugiere que las proporciones del lienzo son transversales a los movimientos artísticos.
  >
  > En **Regresión (P3)**, obtuvimos un Error Absoluto Medio (MAE) de **67.7 años** frente al baseline de la media de 81.6 años. Esto nos indica que las dimensiones geométricas y los colores manuales apenas logran capturar la firma temporal del arte, confirmando la necesidad de implementar Deep Learning en el Hito 2 para extraer texturas e iconografía compleja.
  >
  > Por último, en **Interpretabilidad (P4)**, confirmamos que las variables geométricas son coherentes con la historia del arte: estilos asociados al retrato o arte religioso medieval como el Renacimiento del Norte tienen una alta verticalidad y ratios bajos, mientras que los paisajes del Impresionismo tienden a formatos marcadamente horizontales."*

---

### ⏹ Diapositiva 7: Cierre y Paso a Anexos
* **Tiempo Sugerido:** 15 - 20 segundos.
* **Apoyo Visual:** Título y equipo de cierre.
* **Guión del Orador:**
  > *"Como próximos pasos para el Hito 2, escalaremos la extracción de características visuales a más de 10,000 imágenes e incorporaremos embeddings de alta dimensionalidad utilizando una red ResNet-18 pre-entrenada, además de aplicar Grad-CAM para interpretar visualmente qué zonas de la pintura influyen en la clasificación de estilo.
  > 
  > Con esto concluye nuestra exposición. Quedamos abiertos a sus preguntas y sugerencias. Muchas gracias."*

---

## 🛡️ Guía de Defensa y Preparación para Preguntas (Q&A)

A continuación se presentan las preguntas más probables de la comisión evaluadora (profesores y auxiliares) basadas en los resultados del informe, junto con las respuestas sugeridas y las diapositivas de anexos a las que deben saltar.

### ❓ Pregunta 1: ¿Por qué el rendimiento de la clasificación (F1-Score macro = 0.236) es tan bajo? ¿Cómo piensan solucionarlo en el Hito 2?
* **Cómo responder:**
  > *"El rendimiento es bajo por dos razones principales. Primero, el desbalance de clases en el dataset original (27 géneros con distribuciones muy dispares, dominadas por el Impresionismo y Realismo). Segundo, y más importante, las características visuales manuales actuales (RGB/HSV promedio, brillo, contraste) capturan estadísticas globales de color y luz, pero no formas, trazos de pincel ni estructuras espaciales complejas que caracterizan a los movimientos de arte. 
  > 
  > Para el Hito 2, resolveremos esto reemplazando los descriptores manuales por embeddings profundos extraídos con una **ResNet-18 pre-entrenada** en ImageNet. Esto nos permitirá capturar patrones espaciales y texturas de grano fino. Además, evaluaremos técnicas de sobremuestreo (como SMOTE) o balanceo de pesos en la función de pérdida para manejar el desbalance."*
* **Acción de apoyo:** Ir a **Slide 16** o diapositivas de anexos de clasificación para mostrar la confusión de clases o el código del Random Forest.

---

### ❓ Pregunta 2: El clustering tiene un Silhouette Score muy alto (0.752) en K=4, pero dicen que mezcla todos los estilos artísticos. ¿Significa que el clustering falló o tiene alguna utilidad?
* **Cómo responder:**
  > *"El clustering no falló matemáticamente; de hecho, agrupó los datos de forma sumamente consistente en función de las variables de entrada. El detalle es que las variables utilizadas para el clustering general (del dataset completo) fueron geométricas: `aspect_ratio`, `log_width`, `log_height`, e `is_portrait`. 
  >
  > Lo que el algoritmo encontró son 4 agrupaciones estructurales de lienzos (ej. verticales pequeños, horizontales masivos, cuadrados medianos). El hecho de que todos los clusters tengan una mezcla similar de géneros nos entrega un hallazgo valioso: **los artistas de diferentes épocas y movimientos utilizaron formatos físicos similares**. El formato del lienzo no define el estilo artístico, lo cual justifica por qué debemos enfocarnos en las características de los píxeles (las imágenes en sí) y no solo en los metadatos de las dimensiones físicas en el futuro."*
* **Acción de apoyo:** Ir a las diapositivas de anexos que muestren los gráficos del análisis de clusters (K-means) o PCA (ej. **Slides 10, 11 o 12**).

---

### ❓ Pregunta 3: En el informe se menciona un MAE de regresión de 81.4 años (una mejora del 0.22% sobre el baseline), pero en la Diapositiva 6 de la presentación reportan un MAE de 67.7 años. ¿A qué se debe esta diferencia?
* **Cómo responder:**
  > *"Es una excelente observación. La diferencia radica en los conjuntos de datos evaluados:
  > 1. El valor de **81.4 años** del informe corresponde al Random Forest entrenado sobre el dataset completo de metadatos (47,006 pinturas con año válido), utilizando **únicamente las variables geométricas tabulares** (`aspect_ratio`, dimensiones). Esto confirmó que las dimensiones de los lienzos tienen correlación casi nula con el paso del tiempo.
  > 2. El valor de **67.7 años** reportado en la diapositiva de resultados preliminares corresponde al experimento reducido sobre la muestra de imágenes, donde incorporamos las **características visuales de color y brillo**. Esto demuestra que la información del color (tonos cálidos, niveles de saturación, contrastes cromáticos) sí contiene una firma temporal histórica más fuerte que las simples dimensiones físicas del lienzo, logrando reducir el error absoluto medio en casi 14 años respecto al baseline tabular."*

---

### ❓ Pregunta 4: Extraer características visuales manuales para solo 3,000 imágenes de un total de 80,000 parece una muestra muy pequeña (aprox. 3.75%). ¿Cómo afectará esto la generalización de sus modelos y cómo escalarán el procesamiento de imágenes?
* **Cómo responder:**
  > *"Efectivamente, 3,000 imágenes es una muestra pequeña para generalizar patrones complejos sobre 27 géneros de arte. En este Hito 1, decidimos acotarlo a este número para validar nuestro pipeline de procesamiento cromático de píxeles localmente y asegurar la viabilidad de la extracción matemática de las características de color, brillo y calidez de forma rápida.
  >
  > Para el Hito 2, implementaremos el procesamiento en lotes (batch processing) utilizando PyTorch en GPU para extraer características de forma paralela. Esto nos permitirá escalar la extracción a un mínimo de 10,000 imágenes y, idealmente, a la totalidad del dataset que cuente con imágenes disponibles en los enlaces activos de WikiArt."*

---

### ❓ Pregunta 5: ¿Por qué eligieron árboles de decisión y Random Forest en lugar de modelos más simples como regresión logística o Naive Bayes, o de frentón redes neuronales en este hito?
* **Cómo responder:**
  > *"Elegimos árboles de decisión y Random Forest porque nos proveen una excelente **interpretabilidad** directa. En esta primera fase exploratoria, queríamos entender con precisión qué variables eran las que más aportaban a discernir entre géneros de arte (por ejemplo, el peso geométrico frente al cromático). 
  > 
  > Un árbol de decisión permite extraer reglas de decisión textuales muy claras, y Random Forest nos entrega la importancia de variables (Feature Importance). Esto nos permitió validar hipótesis de la historia del arte, como la marcada relación de verticalidad en retratos renacentistas versus horizontalidad en paisajes impresionistas. Las redes neuronales profundas (CNNs) son cajas negras ideales para el rendimiento, pero los árboles nos sirvieron para cimentar la base empírica e interpretable del problema."*

---

## 📊 Mapeo Estratégico de Anexos (Slides 8 a 16)
Si la comisión hace preguntas de código o visualizaciones detalladas, usen esta guía de referencia rápida para proyectar el anexo adecuado:
* **Si preguntan por el Código de Carga y Limpieza (Multietiquetas/Años):** Proyectar **Slides 8 y 9**.
* **Si preguntan por el Código o Gráficos de Clustering (K-Means y Silhouette):** Proyectar **Slides 10, 11 o 12**.
* **Si preguntan por la Distribución Temporal o Regresión (Q3):** Proyectar **Slide 13 o 14**.
* **Si preguntan por la Importancia de Variables en la Clasificación (Q1/Q4):** Proyectar **Slide 15 o 16**.
