# Registro de decisiones de diseño

Este documento es el registro central del proyecto: recoge cada decisión de diseño del sistema RAG, de principio a fin, junto con la evidencia que la respalda, la fuerza de esa evidencia y su estado. Cada decisión se ata a uno de los objetivos del Trabajo de Fin de Grado (los "OE", objetivos específicos, numerados OE-01 a OE-08 en la memoria), de modo que el capítulo de experimentos de la memoria es, en esencia, el reflejo redactado de lo que aquí esté cerrado con evidencia.

La separación de responsabilidades con otros documentos es intencionada: aquí se registra el *porqué* de cada decisión, con su evidencia; el *cómo está construido* (el detalle de ingeniería) vive en `decisiones_tecnicas.md`.

## Cómo leer este registro

Cada fase corresponde a una etapa del sistema (corpus, representación, índice, recuperación, modelo de embeddings, generación, evaluación y operación). Dentro de cada fase, una tabla lista las decisiones con cuatro columnas: la decisión, las opciones que se barajaron, la evidencia y su nivel, y el estado con el veredicto. Cuando un resultado es extenso, se desarrolla en prosa debajo de la tabla.

**Niveles de evidencia.** No toda decisión se justifica igual, y conviene ser explícito sobre con qué fuerza se sostiene cada una:

- **N1 — experimento controlado con estadística.** Una comparación reproducible sobre el banco de pruebas validado, con intervalos de confianza (bootstrap) y, cuando procede, contraste pareado. Se reserva para lo central y para lo que es barato de medir.
- **N2 — comprobación empírica ligera.** Una medición o contraste puntual, sin intervalo de confianza formal: una señal, no una prueba. Sirve para descartar opciones malas o confirmar una intuición sin gastar el presupuesto de un experimento N1.
- **N3 — literatura y razonamiento de principios.** Se usa cuando el experimento es inviable por recursos o cuando su resultado es conocido de antemano y medirlo sería un gesto vacío. Se justifica citando una fuente fiable. Es evidencia legítima en un trabajo de este tipo; lo que no sería honesto es disfrazarla de N1.

La regla de fondo es que "justificado" no quiere decir "haber ablacionado todo". Cada decisión lleva la evidencia más fuerte que resulte factible, y se declara su nivel sin maquillarlo.

**Estados.** Para no depender de iconos, el estado de cada decisión se indica con una etiqueta entre corchetes:

- **[Cerrada]** — hay evidencia suficiente para defenderla.
- **[Por principio]** — decidida por literatura o razonamiento (N3); no se va a experimentar, y se explica por qué.
- **[Pendiente]** — experimento planificado, con su diseño, todavía sin ejecutar.
- **[Preliminar]** — hay resultado, pero sobre el corpus de 10 normas o una muestra insuficiente; conviene re-medir.
- **[Abierta]** — sin decidir.

---

## Fase 0 — Pregunta de investigación y marco

| Decisión | Evidencia (nivel) | Estado y veredicto |
|---|---|---|
| Construir un sistema RAG informativo sobre legislación consolidada del BOE, de principio a fin, con el énfasis puesto en medir con evidencia qué decisiones producen respuestas más fiables | N3 (marco del trabajo) | [Cerrada] Objetivo general; la interfaz web y la API quedan como trabajo futuro |
| Tomar como experimento central la comparación de recuperadores (denso frente a BM25 frente a híbrido), por ser donde hay una pregunta abierta con respuesta medible y defendible sin depender del juez automático | N3, que pasa a N1 al ejecutarlo | [Cerrada] como marco; ejecutado en la Fase 4 |

---

## Fase 1 — Corpus e ingesta (OE-01)

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Usar la API oficial del BOE de legislación consolidada como fuente | API oficial / scraping de HTML | N3: la API oficial ofrece identificador europeo de legislación, metadatos, versiones y bloques, lo que da trazabilidad estructural | [Cerrada] |
| Guardar el contenido descargado tal cual (inmutable) con su huella sha256 antes de parsear | guardar el original / parsear al vuelo | N3: reproducibilidad y separación entre ingesta y parsing | [Cerrada] |
| Incluir solo normas vigentes, en estado "Finalizado" y con todos los endpoints obligatorios | — | N3: evita indexar texto no consolidado o derogado | [Cerrada] |
| Fijar el tamaño del corpus en 92 normas (partiendo de 10 en el prototipo), cubriendo doce áreas de alto uso ciudadano, con los identificadores verificados en boe.es | 10 / unas 50 / 92 / 100 | N3: selección razonada por áreas y variedad estructural. No es un experimento, sino una decisión de cobertura | [Cerrada] (92 normas; la Ley 35/2015 se descartó por no tener texto consolidado propio) |
| Diferir el Código Civil y la Ley Orgánica del Poder Judicial | incluir / diferir | N3: su formato y tamaño suponen un riesgo para el parser que no compensa el valor marginal con la entrega cerca | [Cerrada] (diferidos a trabajo futuro) |

Referencias de literatura: el identificador europeo de legislación (ELI) y la API de legislación consolidada de la Agencia Estatal BOE.

---

## Fase 2 — Representación documental y troceado (OE-02)

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Usar cuatro artículos de datos con propiedad única del texto (descriptor, historial, bloques padre y fragmentos) | monolítico / compuesto | N3: evita duplicar el texto, se resuelve por unión de identificadores y la eficiencia es medible (los fragmentos ocupan 13 MB sin replicar el texto) | [Cerrada] |
| Decidir la versión vigente por la fecha del índice, no por el orden en el XML | orden del XML / fecha del índice | N3 (regla del dominio) más verificación por una auditoría independiente. Es una cuestión de corrección, no de preferencia: con el criterio antiguo, cuatro bloques servían texto histórico | [Cerrada] |
| Trocear por párrafos respetando la jerarquía (el bloque jurídico actúa como padre del fragmento) | tamaño fijo por caracteres / por párrafos / semántico | N3: respeta la unidad jurídica; alineado con la literatura de recuperación padre-hijo | [Cerrada] (como estrategia) |
| Limitar cada fragmento a 1800 caracteres | 512 / 1024 / 1800 / 2048 | Heredado del prototipo, sin ablacionar | [Preliminar] Candidato a una comprobación N2: barrido del límite frente a la métrica de recuperación |
| Solapar un párrafo solo al dividir un bloque que no cabe | sin solape / varios párrafos | N3: continuidad sin duplicar bloques que ya caben enteros | [Cerrada] (revisable si el barrido del límite lo sugiere) |
| Anteponer el contexto (norma, jerarquía y título) al texto de recuperación | texto crudo / texto con contexto | N3 y literatura de recuperación contextual: el contexto mejora la recuperación de pasajes | [Cerrada]; cuantificable comparando las dos vistas de fragmento |
| Mantener tres vistas del fragmento: con contexto, texto crudo y ventanas de tamaño fijo | — | El prototipo eligió la vista con contexto sobre el corpus de 10 normas | [Pendiente] Re-confirmar con contexto frente a texto crudo en el modelo ganador sobre las 92 normas |
| Linealizar por filas las tablas codificadas como celdas crudas | descartar / linealizar | N3 (corrección: recuperaba texto que antes se perdía), validado por la auditoría | [Cerrada] |

Referencias de literatura: troceado padre-hijo (de lo pequeño a lo grande) y recuperación contextual.

---

## Fase 3 — Almacenamiento e índice vectorial (OE-02/03)

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Usar un índice exacto (producto escalar sobre vectores normalizados, leídos por memoria mapeada) en lugar de un índice aproximado o una base de datos vectorial | exacto / aproximado / base de datos vectorial | N3 (por principio): con unos 25.000 vectores, el índice exacto da recuperación perfecta en menos de un milisegundo y con memoria trivial. Los índices aproximados cambian exactitud por velocidad y solo compensan a partir de millones de vectores; medirlo aquí no aportaría nada | [Cerrada] (exacto); los índices aproximados quedan como trabajo futuro a otra escala |
| Guardar los embeddings exactos, sin cuantizar | exactos / cuantizados | N3: mismo argumento de escala | [Cerrada] (exactos) |
| Publicar el índice como un paquete inmutable con verificaciones (preparación, validación, sumas de control, manifiesto y publicación atómica) | — | N3: reproducibilidad | [Cerrada] |

Referencias de literatura: los métodos de índice aproximado (FAISS, HNSW) tratan precisamente el problema de escala que aquí no se da.

---

## Fase 4 — Recuperación: el experimento central (OE-03/04)

Esta es la pregunta de investigación principal del trabajo: con un buen modelo de embeddings sobre legislación consolidada, ¿conviene la recuperación densa, la léxica (BM25) o una combinación de ambas?

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Comparar recuperación densa, BM25 e híbrida | tres estrategias | N1, el experimento central: métrica de recuperación con intervalo de confianza por bootstrap, contraste pareado frente a la densa y desglose por tipo de pregunta; ajustado en desarrollo y reportado en el conjunto de prueba reservado | [Cerrada] (prueba, n=28, re-puntuado sobre el gold definitivo `corpus92_v1`). La densa gana con 0.806 [0.711, 0.885]; la fusión ponderada con peso alto al denso queda en 0.779 (diferencia pareada −0.027, IC95 [−0.071, +0.020], no significativa, p=0.29); el RRF en 0.691 (−0.115, IC95 [−0.213, −0.027], p=0.006, significativamente peor y **sobrevive a Holm**, p_Holm=0.012); BM25 sola en 0.521 (−0.285, IC95 [−0.410, −0.157], p_Holm<0.001, significativamente peor). Robustez dev+test (n=81) confirma el orden y acota la ganancia posible de la fusión ponderada a +0.016. El sistema usa recuperación densa |
| Elegir la función de fusión: por rango (RRF) o convexa por puntuación normalizada | RRF / convexa / peso α | N1 | [Cerrada] En desarrollo y prueba, la convexa con peso alto al denso supera a la fusión por rango; el RRF es significativamente peor que la densa sola. La mejor fusión no mejora de forma detectable a la densa: la diferencia pareada denso−convexa es −0.028 con IC95 [−0.075, +0.019] (contiene el 0), así que la mejora de la fusión, de existir, está acotada por arriba a +0.019 nDCG. No es equivalencia demostrada, sino ausencia de mejora detectable con n=28 |
| Ajustar BM25 (stopwords, lematización, conservación de cifras, parámetros k1 y b) | ablación de un factor cada vez | N1 | [Cerrada] (desarrollo, n=53, report `bm25abl_20260624T233009Z`). El único parámetro con efecto significativo es el refuerzo de cabecera (fila siguiente); la lematización ayuda (quitarla resta 0.056); stopwords, k1 y b son indiferentes |
| Reforzar la cabecera del bloque en BM25 (repetir el título y la ley a la que pertenece) | sin refuerzo / con refuerzo | N1 | [Cerrada] El refuerzo aporta entre 0.029 y 0.048 según su intensidad, todos significativos. El efecto es grande en las preguntas por número de artículo, donde sube de 0.18 a 0.35: incluir la ley deshace la confusión entre artículos con el mismo número en normas distintas |
| Fijar la profundidad de recuperación que pasa a la generación | 3 / 5 / 10 / 20 | N1 barato (curva de la métrica frente a la profundidad) | [Pendiente] |
| Reordenar resultados con un modelo cruzado (reranking) | sin / con | N3 por ahora: un modelo cruzado en CPU es caro por consulta | [Abierta] (probablemente un piloto pequeño o trabajo futuro) |

Referencias de literatura: BM25/Okapi; la fusión por rango RRF; la comparación entre recuperación densa y léxica del banco BEIR; y MTEB para la selección de modelo.

### Por qué la fusión no mejora aquí

El resultado refuta lo que se había observado en el corpus de 10 normas, donde el híbrido rescataba las preguntas por número de artículo. Con 92 normas, la colisión de números de artículo entre leyes distintas dispara los falsos positivos de BM25, que recupera el artículo correcto pero también muchos de otras normas. El re-pooling lo confirmó: de 138 candidatos aportados por BM25, 127 resultaron irrelevantes. La complementariedad entre la señal léxica y la densa es, por tanto, débil en este corpus, y un modelo de embeddings fuerte con instrucción no necesita la ayuda léxica. Es un hallazgo honesto y ligado al tamaño del corpus: más normas implican más colisión léxica.

### Diseño del experimento

BM25 y el híbrido están implementados y probados sobre las mismas filas que el índice denso, de modo que la comparación es directa. El analizador en español normaliza el texto, aplica stopwords y lematización, y conserva las cifras (por ejemplo, "40.000" o "1,5"). La coincidencia se decide por solape léxico real y no por el signo de la puntuación, lo que evita un problema conocido de BM25 en corpus pequeños.

Antes de comparar fue necesario un paso contra el sesgo: el banco de pruebas se había construido agrupando candidatos solo de los tres recuperadores densos, así que los artículos que solo BM25 encuentra no estaban juzgados y habrían contado como irrelevantes, penalizando a BM25 e híbrido justo en su punto fuerte. Por eso se re-agruparon los candidatos añadiendo BM25 e híbrido y se juzgaron los nuevos. Es el sesgo de agrupación clásico de la evaluación de recuperación.

El orden de ejecución fue: re-agrupar candidatos, ajustar BM25 en desarrollo, ajustar la fusión en desarrollo, calcular la curva por profundidad y, por último, comparar en el conjunto de prueba reservado. Esa última comparación cierra el OE-04.

Referencias de literatura del experimento: Askari et al. (2021) sobre BM25 como base fuerte en el ámbito legal; Bruch et al. (2023) sobre funciones de fusión; Cormack et al. (2009) sobre RRF; Robertson y Zaragoza (2009) sobre BM25; y Thakur et al. (2021) sobre el banco BEIR.

---

## Fase 5 — Modelo de embeddings (OE-02/03)

> **Fuente de verdad = la memoria** (`thesis/`, capítulo de experimentación), cuyas tablas están
> re-puntuadas sobre el gold definitivo `corpus92_v1` (567 juicios: 96 centrales, 75 de apoyo, 396
> negativos, todos revisados). Los titulares de abajo se han sincronizado con ella. Los cortes de
> detalle por estilo y dificultad en *development* corresponden a la corrida de **selección de modelo**
> y pueden diferir en ≤0,01 de las cifras re-puntuadas de la memoria; el análisis por estilo vigente y
> con potencia es el de *dev*+*test* (n=81) del experimento central. Ante cualquier discrepancia,
> prevalece la memoria.

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Usar `e5-large-instruct` como modelo de embeddings | comparativa de tres: `e5-large-instruct`, `bge-m3`, `e5-base` | N1 sobre el banco de las 92 normas (n=53): métrica de recuperación con intervalo de confianza, contraste pareado, frontera de Pareto y cortes por tipo de pregunta | [Cerrada] (2026-06-24). 0.795 frente a 0.717 de `bge-m3` (diferencia pareada −0.078, significativa) y 0.622 de `e5-base` (−0.173, significativa). Es Pareto-óptimo: `bge-m3` queda dominado, con peor calidad y un 47 % más de latencia. Detalle más abajo |
| Usar la instrucción de consulta orientada a lo jurídico | genérica / jurídica / ciudadana | N1 (la misma comparativa) | [Cerrada] La instrucción jurídica (0.795) no muestra diferencia detectable con la ciudadana (0.792) y supera con claridad a la genérica (0.738, diferencia −0.057 significativa). La instrucción de consulta es, por sí sola, una palanca real de mejora |
| Descartar otros modelos con evidencia, no por intuición | `qwen3-0.6b`, `gte-multilingual-base`, `e5-large` | N3 documentado | [Cerrada] `qwen3` tiene un coste prohibitivo en CPU (un hallazgo en sí para un sistema sin GPU); `gte` es incompatible con la versión de la librería de transformers y queda aplazado; `e5-large` es redundante con su variante con instrucción |
| Fijar la revisión exacta del modelo (su commit) | fijada / rama principal | N3: reproducibilidad; el sistema bloquea la publicación si no está fijada | [Cerrada] |
| Reparar el texto que excede el límite de tokens, sin truncar en silencio | truncar / reparar | N3: corrección, para no perder texto | [Cerrada] |

Referencias de literatura: el banco MTEB; la familia multilingual-E5 y el uso de instrucciones de consulta.

### Resultado de la comparativa de modelos

El informe completo está en `data/processed/reports/dense/benchmarks/bench_20260624T181000Z/`. No se versiona, por la política de no guardar artefactos generados, pero se regenera de forma determinista (semilla fija) con el banco de pruebas y los paquetes de índice. La comparación se hizo sobre el conjunto de desarrollo (53 preguntas).

| Modelo y perfil | Métrica | Intervalo de confianza | Frente a la referencia (pareado) |
|---|---|---|---|
| e5-large-instruct, instrucción jurídica | 0.795 | [0.705, 0.873] | referencia |
| e5-large-instruct, instrucción ciudadana | 0.792 | [0.704, 0.868] | −0.003, no significativo |
| e5-large-instruct, instrucción genérica | 0.738 | [0.649, 0.814] | −0.057, significativo |
| bge-m3 | 0.717 | [0.630, 0.793] | −0.078, significativo |
| e5-base | 0.622 | [0.523, 0.720] | −0.173, significativo |

En la frontera de calidad frente a latencia quedan `e5-large-instruct` (0.795, 274 ms por consulta) y `e5-base` (0.622, el más rápido con 133 ms); `bge-m3` queda fuera de la frontera, por ser peor y el más lento.

El corte por tipo de pregunta explica por qué tenía sentido el experimento de recuperación de la Fase 4:

| Tipo de pregunta (n) | bge-m3 | e5-base | e5-large-instruct |
|---|---|---|---|
| ciudadana (15) | 0.807 | 0.699 | 0.865 |
| comparativa (6) | 0.683 | 0.716 | 0.806 |
| conceptual (7) | 0.776 | 0.916 | 0.920 |
| por número de artículo (11) | 0.430 | 0.284 | 0.603 |
| léxica (9) | 0.821 | 0.608 | 0.778 |
| procedimental (5) | 0.869 | 0.688 | 0.921 |

Las preguntas por número de artículo se hunden en todos los modelos densos (el ganador baja de 0.80 a 0.60): la recuperación densa no casa bien los números de artículo. Esa es la justificación cuantitativa de haber probado BM25 e híbrido en la Fase 4. La única grieta donde otro modelo supera al ganador es en las preguntas léxicas, donde `bge-m3` (0.821) queda algo por encima.

Por dificultad, el ganador obtiene 0.696 en las fáciles, 0.818 en las medias y 0.837 en las difíciles. La inversión aparente (las fáciles puntúan peor) se explica porque las preguntas marcadas como fáciles para una persona son a menudo búsquedas de un artículo concreto, que son precisamente las difíciles para la recuperación densa.

Sobre la abstención por confianza de recuperación (separar preguntas dentro del corpus de las que quedan fuera), el ganador separa razonablemente bien (área bajo la curva 0.970), mejor que `bge-m3` (0.856) y `e5-base` (0.798). Pero no es perfecto: al umbral elegido, alrededor del 7 % de las preguntas fuera de corpus se cuelan y un 8 % de las respondibles se abstienen de más. Es una señal complementaria, no un sustituto de la abstención en la generación.

Conviene ser claro con los límites: la comparativa de modelos se hizo sobre el conjunto de desarrollo (n=53), que es el conjunto correcto para seleccionar modelo; la significación viene del contraste pareado pese a que los intervalos marginales se solapen. El modelo elegido (`e5-large-instruct` con instrucción jurídica) se valida después en el conjunto de prueba reservado dentro del experimento de recuperación de la Fase 4 (ParentnDCG@10 0.806, n=28); la comparativa a tres bandas no se re-ejecutó en prueba. Además, la comparativa se hizo sobre la vista de fragmento con contexto; comparar esa vista con la del texto crudo en el modelo ganador queda pendiente.

---

## Fase 6 — Generación fundamentada (OE-05)

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Usar un generador local | `qwen2.5:7b-instruct` frente a otros | Local por requisito (modelo de pesos abiertos, sin coste); efecto del tamaño evaluado (7B vs 14B) | [Cerrada] sistema operativo `qwen2.5:7b`; el 14B se explora en la evaluación (§tamaño del generador) y queda como candidato con *gate* de suficiencia |
| Diseñar un prompt restrictivo, a prueba de fallos, con abstención automática cuando no hay evidencia | — | N3 (diseño anti-alucinación); el contraste de prompts depende del juez | [Cerrada] como diseño |
| Elegir cómo se ensambla el contexto (estrategia, presupuesto y número de fragmentos) | — | N2, después confirmado en generación | [Cerrada] Detalle más abajo. Configuración fijada: expansión acotada, presupuesto de 4000 caracteres y tres fragmentos |
| Tomar las citas y los enlaces del corpus, nunca del modelo, y validar los identificadores citados | — | N3 (corrección y trazabilidad) | [Cerrada] |
| Decidir la abstención por umbral de confianza de recuperación | — | N1/N2 piloto | [Preliminar] (señal complementaria) |
| Abstenerse de responder cuando la pregunta cae fuera del corpus (a prueba de fallos) | — | N1 (no necesita juez) | [Cerrada] en el conjunto de prueba reservado. Detalle más abajo |

### El ensamblado de contexto

La pregunta era cuánto contexto pasar al generador y cómo. Se midió la cobertura de la evidencia y una métrica propia de señal frente a ruido dentro del contexto. La expansión completa quedó descartada (igual cobertura, la mitad de densidad y el doble de texto), y pasar solo el fragmento perdía cobertura. Más de cinco fragmentos era pura dilución. La confirmación la dio la propia generación: con tres fragmentos las respuestas salen algo más densas que con cinco —más hechos clave (0.67 frente a 0.60) y citas ligeramente mejores (0.90 frente a 0.88)—, aunque con cinco el sistema responde a alguna pregunta más (36 frente a 31 de 53). Es un compromiso fino, no un desplome: se prima la densidad de la respuesta y el menor coste. Medido con cuidado sobre las 92 normas el efecto es moderado, lejos del hundimiento que sugería el prototipo de diez normas (donde caía de 0.74 a 0.57) por el conocido efecto de "información perdida en medio". La configuración quedó fijada en expansión acotada, 4000 caracteres y tres fragmentos.

### Seguridad frente a utilidad, y por qué el sistema es conservador

La decisión más delicada en un sistema legal es cuándo callar. El sistema admite dos prompts: uno conservador y uno permisivo. El recorrido fue el siguiente.

Con el prompt conservador, sobre las preguntas fuera de corpus el sistema no dio ninguna respuesta indebida en las 30 de dominio lejano (0/30) y solo una en las 10 near-miss (1/10) —una respuesta indebida en 40—; no incluyó ningún dato prohibido, y la auditoría manual de las 17 respuestas que sí dio no encontró invenciones de dato. El precio de esa prudencia es una sobre-abstención del 39,3 % (responde al 61 % de lo respondible).

El prompt permisivo se probó para recuperar utilidad. Bajó la sobre-abstención del 34 % al 9 %, pero rompió la garantía de seguridad: pasó a responder al 30 % de las preguntas fuera de corpus en lugar de abstenerse. Es un compromiso entre seguridad y utilidad que el prompt por sí solo no resuelve: el modelo de 7000 millones de parámetros no respeta de forma fiable ni su propia instrucción de abstenerse. Por eso el sistema operativo usa el prompt conservador: en el ámbito legal, no dar respuestas falsas es innegociable.

La salida de fondo, documentada como trabajo futuro, es una compuerta de abstención basada en la confianza de recuperación combinada con un prompt más permisivo para las preguntas dentro del corpus. La geometría de las puntuaciones, sin embargo, está apretada (las de dentro de corpus se solapan con las de fuera en torno a 0.90), así que esa compuerta mejoraría la frontera pero dejaría un error irreducible.

El análisis de errores de las respuestas (documento `analisis_errores_generacion.md`) detalla los dos patrones de fallo: responder desde el artículo equivocado por una colisión en la recuperación, y la sobre-extensión del modelo, que añade detalle no respaldado por la evidencia. Con esto, el OE-05 queda cerrado en el conjunto de prueba reservado: el sistema es seguro y fundamentado, a costa de ser conservador. Queda un fallo menor pendiente, la fuga ocasional de los identificadores internos de evidencia al texto de la respuesta.

---

## Fase 7 — Evaluación y juez automático (OE-06/07)

| Decisión | Opciones | Evidencia (nivel) | Estado y veredicto |
|---|---|---|---|
| Evaluar en seis capas (recuperación, contexto, fidelidad, citas, corrección y abstención) | — | N3 (diseño que localiza dónde falla el sistema) | [Cerrada] |
| Tomar como métrica primaria de recuperación el nDCG@10, con controles | — | N3 y literatura (es la métrica estándar) | [Cerrada] |
| Usar como juez automático un modelo de familia distinta a la del generador, para evitar el sesgo de auto-preferencia | misma familia / familia distinta | N3 y literatura sobre el sesgo de auto-preferencia en jueces basados en modelos de lenguaje | [Cerrada] la elección; su validez, más abajo |
| Validar el juez contra anotación humana, con varios coeficientes de acuerdo | fiarse / validar | N1, un entregable comprometido del trabajo | [Cerrada] El juez se validó y se halló insuficiente. Detalle más abajo |
| Construir un banco de relevancia graduado para las 92 normas, con evidencia por párrafo y revisión humana | — | El verdadero desbloqueo de la evaluación | [Cerrada] banco `corpus92_v1`: 121 preguntas y 567 juicios (96 rel-2 / 75 rel-1 / 396 rel-0), todos revisados; párrafos exactos en los centrales |
| Estratificar por tipo de pregunta y dificultad, con contraste pareado, frontera de Pareto y curva de abstención | — | N1 (maquinaria lista y probada) | [Cerrada] implementado |

Referencias de literatura: el uso de modelos de lenguaje como juez y sus sesgos (Zheng et al., 2023); el coeficiente AC1 de Gwet (2008) y la paradoja de prevalencia del coeficiente kappa (Feinstein y Cicchetti, 1990); y los marcos de evaluación de RAG (RAGAS, ARES).

### Por qué el juez automático no se usa como métrica

El objetivo OE-06 pedía validar el juez antes de fiarse de él, y eso es justo lo que se hizo. El resultado es que el juez sobre-acredita en los dos ejes que debía medir, así que sus puntuaciones de fidelidad y corrección no se citan como métrica de calidad del sistema; en su lugar, esas dos dimensiones se reportan a partir de la anotación humana del autor, y la limitación del juez se documenta como un resultado en sí mismo.

Los números, sobre la anotación humana (30 respuestas para fidelidad, 15 para corrección): en corrección, el acuerdo fue del 67 %, con un kappa ponderado de 0.21 y un coeficiente AC1 de 0.60 (en el límite); los cinco desacuerdos fueron todos al alza, es decir, el juez califica mejor de lo que merece. En fidelidad, el acuerdo fue del 87 %, con kappa de 0.44 y AC1 de 0.83, pero el juez solo detectó 2 de las 6 respuestas no fieles: cuatro invenciones pasaron como fieles, que es justo la dirección peligrosa en el ámbito legal.

Se intentó calibrar el prompt del juez (una tercera versión). La corrección mejoró algo (AC1 de 0.60 a 0.67), pero la fidelidad empeoró, y lo decisivo es que las mismas cuatro invenciones pasaron desapercibidas tanto con el prompt permisivo como con el escéptico. Eso indica que el fallo es de capacidad del modelo juez, no del prompt, y que seguir iterando el prompt da rendimientos decrecientes. La conclusión es que un juez automático de este tamaño no es fiable para fidelidad y corrección en este dominio; las salidas serían usar un modelo juez más potente y de familia distinta, o reportar la anotación humana, que es lo que se ha hecho.

---

## Fase 8 — Operación, reproducibilidad y uso responsable (OE-08)

| Decisión | Evidencia (nivel) | Estado |
|---|---|---|
| Gestionar el entorno con `uv`, usar los contratos Pydantic como fuente única de verdad y mantener la suite de tests cien por cien sin red | N3 (reproducibilidad y mantenibilidad) | [Cerrada] |
| Ejecutar todo con software de código abierto y modelos de pesos abiertos, en local (CPU), sin servicios de pago | N3 (requisito del trabajo) | [Cerrada] |
| Incluir un aviso jurídico estático: el sistema no sustituye a la publicación oficial | N3 (requisito del trabajo) | [Cerrada] |

---

## Estado final y trabajo futuro

El alcance experimental del TFG está **cerrado**. Hecho: el banco de relevancia de las 92 normas
(graduado, revisado uno a uno, con evidencia por párrafo en los centrales); el experimento de
recuperación completo (denso vs BM25 vs híbrido, ablaciones de BM25 y de fusión, robustez n=81); la
comparativa de modelos de *embeddings* y la ablación de representación (incluida la vista de fragmento
enriquecida frente a la cruda, sin diferencia detectable); y la generación fundamentada con sus
*baselines* (closed-book, oráculo), el efecto del tamaño del generador (7B vs 14B), el contraste de
*prompts* y la validación del juez (resultado negativo, por baja sensibilidad).

Queda como **trabajo futuro**, ya fuera del alcance de la entrega:

- **Reordenador (*cross-encoder*)** sobre el candidato denso — previsto pero no implementado; se cita
  como línea abierta (OE-03).
- **Compuerta de suficiencia** para recortar la sobre-abstención sin perder seguridad, y su
  combinación con un generador mayor (14B + *gate*).
- **Ampliar la anotación humana** de fidelidad/corrección (un solo anotador, n pequeño) y, con ella,
  un juez de mayor capacidad; curva de la métrica por profundidad de recuperación.

No se experimentará, **por decisión de principio y justificado por escala (N3)**, con índices
aproximados frente al exacto ni con *embeddings* cuantizados frente a exactos: a ~25k vectores el
índice exacto es holgado y la ablación sería teatro.

## Cómo se mantiene este registro

Cuando una decisión se cierra con evidencia, se actualiza su fila (estado y enlace al informe correspondiente) y, si procede, se redacta en la memoria. Antes de abrir un experimento conviene comprobar aquí su nivel de evidencia, para no gastar un experimento controlado en algo que se resuelve por principio. El calendario de cuándo hacer cada cosa vive en el plan del proyecto; este registro guarda qué evidencia respalda cada decisión y por qué.
