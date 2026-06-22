# RESULTADOS — registro de experimentos (evidencia)

> Resultados del **MVP cerrado** (corpus limpio de 10 normas + gold de relevancia **validado**, Fase D).
> Generado el **2026-06-19** leyendo el bake-off denso (`dense/bench_20260619T063600Z`) y el flagship
> denso/BM25/híbrido (`dense/benchmarks/retrieval_20260619T084908Z`).
> Contexto estable: `CLAUDE.md` · norte del TFG: `PLAN.md` · plan de cierre: `CIERRE_MVP.md`.

## 0 · Validez (LEER PRIMERO)

> ✅ **POST-FIX.** El bake-off de §1 está medido sobre el **corpus reprocesado** (parser endurecido:
> aparato editorial `<blockquote>` + tablas forma A/B linealizadas) y el **gold de relevancia
> revisado** (Fase D: relevancia graduada 2/1/0, evidencia por párrafo, multi_parent, trampas).
>
> ⚠️ **VOID (superado):** todos los números **pre-fix** medidos sobre el corpus sucio + gold borrador
> (el bake-off del 06-jun, la comparación BM25/denso/híbrido del 13-jun y la generación/κ del juez)
> quedan **invalidados** y NO se citan. La generación (L3–L6) y la κ del juez **no se han re-medido**
> sobre el corpus limpio todavía (Fase F, apoyo secundario; el gold de generación `answer_keys` sigue
> `draft`).

- Medido sobre: **corpus 10 normas limpio · split `development` (n=50) · gold reviewed** (Gate C
  checkpoint: dev ✅, ooc ✅, test 19/20 — la trampa q0051 no cuenta, ver `CIERRE_MVP`).
- Métrica primaria: **`ParentnDCG@10`**, IC bootstrap (1000 resamples, seed 12345), diferencias
  pareadas vs baseline. Las afirmaciones marcan **[SÓLIDO]** / **[PRELIMINAR]**.
- Datos: los reports están en `data/processed/reports/dense/bench_20260619T063600Z` (bake-off) y
  `data/processed/reports/dense/benchmarks/retrieval_20260619T084908Z` (flagship) — gitignored, en
  `dslab01`; el gold sí se versiona en `data/evaluation/dense_retrieval_v1/`.
- El **flagship** BM25/denso/híbrido (§2) ya está medido **post-fix** sobre el corpus limpio + gold
  validado (deja obsoleta la comparación pre-fix del 13-jun, void).

---

## 1 · Bake-off del recuperador denso (MVP) — `bench_20260619T063600Z`

5 modelos (e5-base, e5-large, e5-large-instruct, bge-m3, qwen3-0.6b) × perfiles de query, chunking J1,
sobre el gold validado. `gte` aplazado (incompatible con el stack, ver `docs/known_issues.md`).

### 1a · Calidad por modelo × perfil (`ParentnDCG@10`, 50 dev)

| Modelo · perfil | nDCG@10 | Hit@1 | IC 95% |
|---|---|---|---|
| **e5-large-instruct · I2_CITIZEN** | **0.859** | 0.84 | [0.779, 0.920] |
| e5-large-instruct · I1_LEGAL *(baseline)* | 0.841 | 0.78 | [0.762, 0.907] |
| qwen3-0.6b · I0_GENERIC | 0.821 | 0.80 | [0.738, 0.897] |
| bge-m3 · BASELINE | 0.816 | 0.74 | [0.733, 0.886] |
| qwen3-0.6b · I1 / I_MINUS_NONE | 0.816 | 0.78–0.80 | … |
| e5-large-instruct · I0_GENERIC | 0.810 | 0.76 | [0.727, 0.884] |
| e5-large · BASELINE | 0.804 | 0.76 | [0.725, 0.869] |
| qwen3-0.6b · I2 | 0.796 | 0.76 | … |
| **e5-base · BASELINE** | **0.778** | 0.70 | [0.705, 0.848] |

### 1b · Pareado vs baseline (`e5-large-instruct · I1_LEGAL`)

- **e5-base − baseline = −0.064** [−0.122, −0.007] → **significativamente PEOR** (IC no cruza 0).
- **e5-large-instruct·I2 − baseline = +0.018** [−0.002, 0.041] → **no significativo** (roza, cruza 0).
- Resto (e5-large, bge-m3, qwen3·*): todos con IC que **cruza 0** → sin diferencia significativa.

**[PRELIMINAR] El ranking sugiere `e5-large-instruct·I2`, pero con n=50 los IC se solapan: nadie
supera al baseline con significancia; el único claramente inferior es `e5-base`.** El perfil ciudadano
(I2) es el nominal mejor (coherente con el checkpoint previo). qwen3 prefiere **sin** instrucción.

### 1c · Cortes por estilo de pregunta — **EL HALLAZGO CLAVE**

`ParentnDCG@10` aproximado, consistente en los 5 modelos:

| estilo | nDCG | Hit@1 | lectura |
|---|---|---|---|
| ciudadana / conceptual / procedimental | ~0.78–0.88 | alto | terreno del denso |
| **lexica** (BDNS, ICIO, Portal…) | ~0.82–0.88 | bueno | los términos están literales → el denso los embebe bien |
| comparativa *(n=3)* | ~0.95–0.99 | 1.0 | alto pero poco fiable (n bajo) |
| **directa_articulo** ("¿qué dice el art. 122?") *(n=4)* | **~0.31–0.41** | **0.0–0.25** | **el denso se hunde** |

**[SÓLIDO, cualitativo] El denso falla las consultas de artículo exacto** (~0.35 vs ~0.85 del resto),
de forma **universal en los 5 modelos**: devuelve el preámbulo/disposiciones finales en lugar del
artículo citado (p. ej. art. 38 Ley 40/2015 = miss total del top-20; art. 122 LPAC = 1 sistema, rank 3).
La **léxica NO es el punto débil** (los términos aparecen literales en el texto). **Esto es,
precisamente, el hueco que el BM25/híbrido del flagship debe cerrar** (coincidencia exacta de token
"artículo 122") → **confirmado en §2c: la fusión lo cierra (0.41 → 0.75)**. *(Estrato n=4 →
cuantitativamente preliminar, pero la brecha es enorme y consistente.)*

### 1d · Cortes por dificultad

`facil ≈ media > dificil` (p. ej. e5-large-instruct·I0: fácil 0.911 · media 0.777 · difícil 0.729).
Esperable; la dificultad anotada es coherente con el rendimiento.

### 1e · Abstención (L6) — ¿separa el score las OOC de las respondibles?

Score top-1 del retriever, in-corpus (dev) vs out_of_corpus (10):

| modelo · perfil | ROC-AUC | balanced acc | umbral | TPR | TNR |
|---|---|---|---|---|---|
| **e5-large-instruct · I1** | **0.998** | 0.99 | 0.895 | 0.98 | 1.0 |
| e5-large / e5-base | 0.994 | 0.99 | ~0.84 | 0.98 | 1.0 |
| bge-m3 | 0.97 | 0.93 | 0.56 | 0.96 | 0.90 |
| qwen3-0.6b | 0.96–0.99 | 0.93–0.97 | ~0.54–0.68 | 0.86–0.94 | 1.0 |

**[SÓLIDO, cualitativo] Un umbral simple de score separa casi perfectamente respondible vs
out-of-corpus** (AUC ~0.99, TNR=1.0 en la familia e5). **El RAG puede "saber cuándo callar" por
confianza de recuperación.** El umbral es **por modelo** (escalas de score distintas). *(n=10 OOC → la
separación es muy clara pero el IC del umbral es ancho.)*

### 1f · Frontera calidad/coste (latencia de embedding de query p50, CPU)

| modelo · perfil | nDCG@10 | latencia p50 | frontera |
|---|---|---|---|
| e5-base | 0.778 | **92 ms** | ✅ (económico) |
| e5-large | 0.804 | 207 ms | ✅ |
| e5-large-instruct · I1 | 0.841 | 244 ms | ✅ |
| **e5-large-instruct · I2** | **0.859** | 249 ms | ✅ (mejor calidad) |
| qwen3-0.6b · I_MINUS_NONE | 0.816 | 228 ms | ✅ |
| bge-m3 | 0.816 | 389 ms | ❌ dominado |
| qwen3-0.6b · I0/I1/I2 | 0.796–0.821 | 385–415 ms | ❌ dominado |

**[SÓLIDO] Para despliegue CPU-only, `e5-large-instruct` es la mejor calidad/coste; `e5-base` es la
opción económica (3× más rápido, ~8% menos nDCG); `bge-m3` y `qwen3` están DOMINADOS** (más lentos sin
ganar calidad). Confirma que los modelos pesados/exóticos no compensan en CPU (coherente con qwen3
inviable en encoding y gte aplazado).

### ✅ Decisión del MVP (checkpoint)

**`e5-large-instruct` · chunking `J1` · perfil `I2_CITIZEN_LEGISLATION`** → bundle
`e5-large-instruct__j1__42105deb4afe`. Confirma el baseline (nadie lo supera con significancia;
`e5-base` claramente inferior) y es la mejor calidad/coste en CPU.

---

## 2 · FLAGSHIP — BM25 vs denso vs híbrido — `retrieval_20260619T084908Z`

**El núcleo del TFG.** Denso (`e5-large-instruct·I2`, el ganador del bake-off) vs BM25 (stopwords +
stemming, defaults) vs híbrido RRF (k=60), sobre el MISMO bundle/rows y el gold validado. Split
`development` (n=50), `ParentnDCG@10`, IC bootstrap 1000 / seed 12345, pareado vs denso.

### 2a · Agregado global — la hipótesis ingenua "el híbrido gana" es FALSA

| estrategia | nDCG@10 | IC 95% | Δ vs denso (pareado) |
|---|---|---|---|
| **denso** | **0.861** | [0.781, 0.921] | — (baseline) |
| híbrido RRF | 0.841 | [0.772, 0.902] | **−0.020** [−0.079, +0.040] → no significativo |
| BM25 | 0.681 | [0.608, 0.751] | **−0.180** [−0.253, −0.108] → **significativamente PEOR** |

**[SÓLIDO] Globalmente el denso NO es superado por el híbrido** (queda un pelín por debajo, sin
significancia) y **BM25 en solitario es claramente inferior.** Quedándose aquí se concluiría "el
híbrido no aporta" — y sería un error: el agregado **esconde** el comportamiento por estilo (§2b).

### 2b · Cortes por `query_style` — EL HALLAZGO (justifica estratificar)

| estilo | n | denso | BM25 | híbrido | Δ híbrido−denso |
|---|---|---|---|---|---|
| ciudadana | 20 | 0.872 | 0.716 | 0.822 | −0.050 |
| conceptual | 8 | 0.932 | 0.642 | 0.789 | **−0.143** |
| procedimental | 9 | 0.901 | 0.705 | 0.935 | +0.034 |
| comparativa | 3 | 0.996 | 0.723 | 0.879 | −0.117 |
| lexica | 6 | 0.903 | 0.692 | 0.873 | −0.030 |
| **directa_articulo** | 4 | **0.408** | 0.480 | **0.750** | **+0.342** |

**[SÓLIDO, cualitativo] El valor del híbrido es quirúrgico, no global:** rescata el ÚNICO estilo donde
el denso colapsa (`directa_articulo` **0.41 → 0.75**, +0.34) a cambio de pequeñas regresiones en el
resto. Ese +0.34 sobre 4 preguntas queda **diluido** por las regresiones de las otras 46 → de ahí el
−0.02 global. **Sin el corte por estilo el hallazgo es invisible.** Dos matices que corrigen ideas
previas:
- **`lexica` NO es territorio de BM25**: el denso (0.903) **gana** a BM25 (0.692). El modelo embebe
  bien siglas/términos; lo único que se le resiste es la **referencia exacta de artículo**.
- **El héroe de `directa_articulo` no es BM25 solo (0.480) sino la FUSIÓN (0.750)**: ni denso ni BM25
  por separado bastan; combinados, sí. Argumento de manual del híbrido.

### 2c · Las 4 `directa_articulo`, caso a caso

| pregunta | denso | BM25 | híbrido | lectura |
|---|---|---|---|---|
| **q0077** art. 122 LPAC | **0.0** (top1=a127) | 0.63 (a122 en top10, no rank1) | **1.0** (a122 rank1) | ★ complementariedad pura: denso 0, BM25 parcial, **fusión perfecta** |
| q0078 art. 12 L19/2013 | 0.63 (top1=df 8ª) | **1.0** (a12) | **1.0** (a12) | rescate de BM25; la fusión lo mantiene |
| q0076 art. 21 LPAC | **1.0** (a21) | 0.29 (top1=a23) | **1.0** (a21) | la fusión no rompe un acierto del denso (BM25 metía a23) |
| q0079 art. 38 LRJSP | 0.0 | **0.0** (top1 de otra norma) | **0.0** | **sin resolver por nadie** — el caso honesto |

**[SÓLIDO] q0077 es el póster del TFG:** el denso no recupera a122 ni en el top-20 (coge el vecino
a127), BM25 lo encuentra pero no en cabeza, y **RRF lo coloca en el puesto 1**. Demostración limpia de
que denso y léxico capturan señales distintas y la fusión las combina.
**q0079 (art. 38) es el contraejemplo valioso:** falla **incluso BM25** (top1 = una disposición
transitoria de OTRA norma de 2003); "38" es ambiguo en el corpus y el "38" del título no pesa más que
cualquier otro → **confirma con datos la mejora pendiente: ponderar la cabecera/nº de artículo
(BM25F-lite).** No es hipótesis, es el fallo observado (q0079 + los top1 erróneos a23/df-6).

### 2d · Cortes por dificultad

| dificultad | n | denso | BM25 | híbrido |
|---|---|---|---|---|
| facil | 15 | 0.923 | 0.714 | 0.889 |
| media | 27 | 0.839 | 0.669 | 0.841 (empate) |
| dificil | 8 | 0.820 | 0.658 | 0.748 |

El híbrido solo iguala al denso en `media`; el efecto vive en el **estilo** `directa_articulo`, no en
la dificultad.

### 2e · Mejoras del flagship: heading-boost (BM25F-lite) + fusión ponderada

Dos experimentos sobre el mismo bundle/gold (boost = copias EXTRA de la cabecera del bloque al
indexar BM25; α = peso del denso en la fusión ponderada por score min-max). Reports
`retrieval_20260621T13*/17*/18*Z`.

**(i) Heading-boost del nº de artículo.** Boostear SOLO el nº ("Artículo 122") mejora BM25 pero
**rompe el híbrido** en `directa` (0.75→0.658): el mismo número existe en dos leyes (a122 está en la
Ley 39 **y** en la Ley 40); el boost lo amplifica en ambas y ahoga el discriminador de ley → el RRF,
con el denso confundido, elige la ley equivocada (q0077). **Incluir la LEY en el boost**
(`citation.label` + título) lo arregla: BM25 solo pasa a `directa` **1.0** (incl. q0079) y el híbrido
recupera la paridad. El boost satura en **3**.

**(ii) Fusión: RRF (rango) vs ponderada (score).** Con `boost=3`+ley:

| estrategia | global | directa | ciudadana | conceptual | comparativa |
|---|---|---|---|---|---|
| denso | 0.861 | 0.41 | 0.872 | 0.932 | 0.996 |
| BM25 (boost3+ley) | 0.783 | **1.00** | 0.788 | 0.666 | 0.723 |
| híbrido RRF | 0.880 | 0.741 | 0.898 | 0.833 | 0.879 |
| **híbrido ponderado α0.5** | **0.888** | **0.875** | 0.899 | 0.788 | 0.954 |
| híbrido ponderado α0.3 | 0.870 | 0.908 | 0.868 | 0.779 | 0.906 |

**[SÓLIDO, mecanístico] Para la cita exacta, el RRF (basado en RANGO) DILUYE el acierto del léxico; la
ponderada (score normalizado) lo TRANSMITE.** BM25+boost da un score alto y disambiguado por ley al
artículo correcto; el RRF lo aplana a `1/(k+rango)` y deja ganar al denso (confundido entre
homónimos), mientras la ponderada conserva la magnitud. Resultado: la ponderada **arregla q0077** (que
el RRF fallaba) y gana al RRF en `directa` (0.74→0.875) **y** en global (0.880→**0.888**, la mejor de
todas). Per-query ponderada (α0.5): q0076/q0077/q0078 = 1.0; **solo resiste q0079** (art.38, 0.5): a38
entra en el top pero el denso lo ancla en disposiciones finales de la Ley 40 ("sede electrónica") → es
colisión con la **ley derogada**, no homónima → lever de **filtros por ley/fecha**, no de fusión.

### ✅ Config final del flagship (en `dev`)

**denso `e5-large-instruct·I2` + BM25 (`heading-boost=3`, ley+título) + fusión PONDERADA α≈0.5.**
Global **0.888** (la mejor de todas: denso 0.861, RRF 0.880), `directa` **0.875** (denso 0.41 → BM25+ley
1.0 → RRF 0.74 → ponderada 0.875), resto de estilos ≈ RRF o mejor. `α0.3` exprime `directa` a 0.908 a
costa de algo de global (palanca si se prioriza la cita exacta). El código deja `heading_boost=0` /
RRF como baseline reproducible; esta config se fija con flags.

### 2f · Validación en `test` held-out — `retrieval_20260621T182539Z` (n=20)

| estrategia | nDCG@10 | IC 95% | Δ vs denso |
|---|---|---|---|
| **denso** | **0.791** | [0.664, 0.904] | — |
| híbrido ponderado α0.5 | 0.757 | [0.629, 0.870] | −0.034 [−0.114, +0.042] (ns) |
| híbrido RRF | 0.686 | [0.544, 0.823] | −0.105 [−0.249, +0.014] |
| BM25 | 0.539 | [0.387, 0.698] | **−0.252** [−0.417, −0.090] (peor) |

**[SÓLIDO] Lo que generaliza:** el orden **denso ≥ ponderada > RRF > BM25** se replica; en particular
**ponderada > RRF** (0.757 vs 0.686; `procedimental` 0.90 vs 0.68, `conceptual` 0.82 vs 0.79) → el
hallazgo metodológico (fusión por **score** > por **rango**) aguanta en held-out.

**[LÍMITE DEL BANCO] Lo que NO se puede validar aquí:** en `test` el **denso gana** (0.791 vs 0.757,
ns). El triunfo del híbrido en `dev` lo movía `directa_articulo`, y **el split `test` casi no tiene de
ese tipo**: su único `directa_articulo` es **q0051, la trampa** de abstención (todas 0.0 por diseño);
`comparativa` tampoco aparece. Es decir, **el banco held-out no cubre el caso donde el híbrido aporta**,
así que su ventaja no se confirma ni se refuta aquí. El `boost` es **~neutro** en test (sin cita exacta
no actúa) → adoptarlo no penaliza. **→ Convierte el corpus-100 + banco ampliado (con suficientes
`directa_articulo` y `comparativa` en `test`) de deseable a NECESARIO** para cerrar la afirmación.

### ✅ Tesis del flagship

**No "el híbrido es mejor", sino "cuándo conviene cada estrategia".** Con un denso fuerte off-the-shelf
en corpus legal: el **denso domina** salvo en **consultas de artículo exacto**, donde colapsa (~0.41) y
**solo la fusión** lo recupera (~0.75); el coste es una ligera regresión en conceptual/comparativa (el
ranking ruidoso de BM25 contamina). BM25 en solitario nunca es la respuesta en este corpus. Coherente
con la literatura: con modelos **zero-shot** la fusión ayuda **concentrada donde importa el match
léxico exacto**, no de forma uniforme. **El hueco se CIERRA** (§2e) con **boost de cabecera+ley
resuelto por fusión ponderada**: en `dev` lleva `directa_articulo` de 0.41 a 0.875 y el global a 0.888,
superando al denso. **Matiz de held-out (§2f):** en `test` el denso aún gana, porque ese split casi no
tiene `directa_articulo` (es donde el híbrido aporta) → confirmar la ventaja exige el **corpus-100 +
banco ampliado**. Hallazgo transversal que SÍ replica en test: **la fusión por score (ponderada) > la
fusión por rango (RRF)** cuando una vía tiene alta confianza (la cita exacta del léxico).

### Caveats
- **n pequeño:** `directa_articulo` n=4 → IC enorme (híbrido [0.25, 1.0]). Efecto grande y
  mecánicamente explicado, pero **cuantitativamente preliminar** (potencia → corpus-100).
- **Un solo modelo denso** medido. Ablaciones de heading-boost y fusión (α) **hechas** (§2e);
  stopwords/stemming aún sin ablar.
- **Validación en `test` pendiente:** §2e está tuneado en `dev` (n=50) → α=0.5 podría estar
  sobreajustado; confirmar en held-out. q0079 (colisión con ley derogada) → lever de filtros.

---

## 3 · Pendiente (fases siguientes)

- **Mejoras del flagship — HECHAS (§2e):** heading-boost+ley (`boost=3`) + fusión ponderada (α0.5)
  → `directa` 0.41→0.875, global 0.888. Queda: **validar en `test`** (α puede estar sobreajustado);
  **q0079** (ley derogada) con **filtros por ley/fecha**; ablación de stopwords/stemming.
- **Corpus 10 → 100** + banco ampliado (~150–200) con la metodología de Fase D + BM25 en el pooling
  (da potencia estadística a `directa_articulo`).
- **Generación L3–L6 + κ del juez** sobre corpus limpio: a re-medir (apoyo secundario; el gold
  `answer_keys` sigue `draft`).
- **Test held-out:** re-correr bake-off y flagship en `--split test` para la cifra de validación final.

---

## 4 · Síntesis honesta — qué aguanta una defensa HOY

**[SÓLIDO]:**
- **Cuándo conviene cada estrategia (flagship):** el denso domina salvo en `directa_articulo`
  (0.41), donde **solo la fusión** remonta; BM25 en solitario es inferior en global (−0.18). El
  caso q0077 (denso 0 + BM25 parcial → fusión perfecta) demuestra la complementariedad.
- **BM25 con boost de cabecera+ley RESUELVE la cita exacta** (`directa` 0.41→1.0) y **la fusión
  ponderada (por score) > RRF (por rango)** — replica en `test` held-out (0.757 vs 0.686). Hallazgo
  metodológico reutilizable.
- El denso **falla las consultas de artículo exacto**, universal en los 5 modelos del bake-off →
  motivó el híbrido, ahora **confirmado** con datos limpios.
- **Abstención por umbral de score** funciona (AUC ~0.99); el RAG puede abstenerse por confianza.
- En CPU, **e5-large-instruct = mejor calidad/coste**; **bge-m3 y qwen3 dominados**; `e5-base` económico.
- La léxica (siglas/términos) **no** es el punto débil del denso (el denso incluso gana a BM25 ahí).

**[PRELIMINAR] (dirección clara, n insuficiente):**
- "e5-large-instruct·I2 es el mejor modelo" (dentro del ruido vs e5-large/bge-m3/qwen3; solo e5-base
  separa).
- Magnitud exacta del +0.34 del híbrido en `directa_articulo` (estrato n=4, IC ancho).
- **"El híbrido supera al denso en global"**: solo en `dev` (0.888 vs 0.861) y movido por
  `directa_articulo`; en `test` el denso gana (0.791 vs 0.757, ns) porque ese split casi no tiene
  `directa_articulo` (§2f). Pendiente de confirmar con un banco que cubra ese estilo.

**[PENDIENTE] (bloquea conclusiones fuertes):**
- **Corpus 100** + banco ampliado (donde se juega la tesis "cuándo conviene cada estrategia" con
  potencia, y el `test` con suficiente `directa_articulo`/`comparativa`).
- **Generación (L3–L6) + κ del juez sobre corpus limpio: SIN MEDIR** (Fase F abierta; el pre-fix es
  void). Ver §5 y `CIERRE_MVP.md`.

---

## 5 · Inventario y trazabilidad de experimentos (corpus 10)

Todo experimento corrido sobre el corpus limpio + gold validado, para que sea reproducible y
auditable. Reports en `data/processed/reports/dense/benchmarks/` (gitignored, en `dslab01`); el gold
versionado en `data/evaluation/dense_retrieval_v1/`; bundles en `data/indexes/dense/` (gitignored).

### 5a · Retrieval — POST-FIX (válidos)

| report id | experimento | config | → |
|---|---|---|---|
| `bench_20260619T063600Z` | Bake-off denso (5 modelos × perfiles) | dev n=50; gte aplazado | §1 |
| `retrieval_20260619T084908Z` | Flagship denso/BM25/RRF (baseline) | dev; boost=0 | §2a–2d |
| `retrieval_20260621T133933…134315Z` (5) | Barrido heading-boost **solo-número** (0,1,2,3,5) | dev | §2e(i) |
| `retrieval_20260621T174259…174647Z` (5) | Barrido heading-boost **con ley** (0,1,2,3,5) | dev | §2e(i) |
| `retrieval_20260621T180920…181247Z` (4) | Barrido **fusión ponderada** α (0.2–0.5) | dev; boost=3+ley | §2e(ii) |
| `retrieval_20260621T182539Z` | **Validación held-out** (4 estrategias) | **test** n=20; boost=3, α0.5 | §2f |

**Config final retrieval:** `e5-large-instruct·I2` + BM25(`heading-boost=3`, ley) + fusión **ponderada α≈0.5**.

### 5b · Generación (L3–L6) — SIN CORRER sobre el corpus limpio (Fase F abierta)

- **No existe ningún report de generación POST-fix.** La infraestructura está completa (generador
  `qwen2.5:7b`, juez `gemma3:12b`, `run_generation_eval.py`/`validate_judge.py`/`rejudge_correctness.py`),
  pero **no se ha re-corrido** tras el arreglo del parser.
- **Gold de generación** (`answer_keys.jsonl`): 80 entradas, **80/80 `reviewed`** (D2 cerrado
  2026-06-21: cada `reference_answer` verificada contra el texto del corpus, `key_facts` literales,
  citas vigentes; `audit_eval_dataset` "Sin flags"). Gate C generación checkpoint = **test 19/20**
  (trampa q0051, igual que retrieval). Pendiente: spot-check del autor.
- **Juez:** κ corrección **0.302** (n=32, 1ª pasada, **pre-fix → void**) <0.6; **L3 fidelidad n=3** =
  sin medir. **Diagnóstico metodológico (2026-06-22):** ese κ<0.6 es muy probable la **paradoja de
  prevalencia** (clases desbalanceadas, casi todo "correct" → κ se hunde aunque el acuerdo sea alto;
  el κ ponderado ya daba 0.547). **Implementado Gwet's AC1** (robusto a prevalencia) en
  `judge_agreement` + `validate_judge.py` (Track C1): se reporta κ + κ-ponderado + **AC1** + %acuerdo
  + matriz. La validación del juez "de verdad" (con AC1 + n≥50 + rúbrica con few-shot/CoT) se hará
  sobre el **corpus-100**; la 2ª pasada en 10 no se corre (se rehace). Lit.: Gwet AC1 vs Cohen κ,
  G-Eval/LLM-Rubric.
- **Cerrable ahora** (🖥️): `run_generation_eval.py --bundle …42105deb4afe --generator-model
  qwen2.5:7b-instruct --no-judge` (L4/L6/key-fact sólidos) y con `--judge-model gemma3:12b` (L3/L5
  provisionales). **Requiere 🧑:** anotación κ (corrección + L3 n≥20). Versión con potencia → corpus 100.

### 5c · VOID (superado, NO citar)

Todo lo medido **pre-fix** (corpus sucio + gold borrador): bake-off del **06-jun**, comparación
BM25/denso/híbrido del **13-jun**, generación + κ del juez (`gen_20260612T143618Z…`, κ=0.302). Queda
invalidado por la cascada del parser (`CIERRE_MVP.md §0`).
