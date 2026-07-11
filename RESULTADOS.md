# RESULTADOS — síntesis de evidencia (corpus-92)

> **Fuente única de verdad = la memoria** (`thesis/`, capítulo de experimentación) y el ledger
> `docs/decisiones_de_diseno.md` (Fases 4 y 5), donde viven los números exactos, los intervalos de
> confianza y el modo de reproducirlos. Este fichero es un **resumen de titulares** sincronizado con
> la memoria; ante cualquier discrepancia, **prevalece la memoria**. Estado vivo en `PROGRESO.md`.
> El registro histórico del MVP de 10 normas (incluido lo marcado VOID) está archivado en
> `docs/archive/RESULTADOS_mvp_corpus10.md`.
>
> Banco vigente: gold `corpus92_v1` (121 preguntas; 567 juicios = 96 centrales + 75 de apoyo + 396
> negativos, todos revisados uno a uno), de-sesgado por *re-pooling* con denso+BM25+híbrido.

---

## Recuperación (OE-02/03/04) — N1, cerrada

- **Modelo denso (bake-off, *dev* n=53):** ganador **`e5-large-instruct · I1_LEGAL`**
  (ParentnDCG@10 **0,795**) > bge-m3 (0,717) > e5-base (0,622). Bate a bge-m3 en pareado
  (Δ −0,078, SIG) y bge-m3 queda **Pareto-dominado** (peor calidad y ~47 % más lento). La
  **instrucción de *query* es palanca**: I1 sobre I0 genérico Δ −0,057 (SIG); I1 e I2 sin diferencia
  detectable. → **OE-03 cerrada.**
- **Representación/chunking (*dev*):** enriquecer el fragmento con el nombre de la norma **no mejora**;
  el troceado clásico **iguala** al *parent-child* por párrafos (la métrica premia acertar el artículo).
- **Flagship denso vs BM25 vs híbrido (*test* held-out, n=28):** **denso 0,806** > híbrido ponderado
  α0.7 0,779 > híbrido RRF 0,691 > BM25 0,521. Pareados frente al denso (familia corregida por Holm):
  **BM25 −0,285 (SIG peor, p_Holm<0,001)**, **RRF −0,115 (SIG peor, p=0,006, p_Holm=0,012)**,
  **ponderado −0,027 (n.s., p=0,29; cota superior +0,020)**. → **OE-04 cerrada.**
- **Robustez (*dev*+*test*, n=81; no held-out, sesgo optimista pro-híbrido):** confirma el patrón —
  denso 0,805 > ponderado 0,794 (Δ −0,010 n.s., cota **+0,016**) > RRF 0,724 (Δ −0,081, p_Holm=0,004)
  > BM25 0,531 (Δ −0,274, p_Holm<0,001).
- **Tesis OE-04:** con un *embedder* instruct fuerte sobre legislación consolidada del BOE, el **denso
  supera significativamente a BM25 y a la fusión RRF, y ninguna fusión lo supera**; frente al híbrido
  ponderado no hay diferencia concluyente (su ganancia posible es, como mucho, +0,016). Por parsimonia,
  **la fusión no aporta** y el sistema usa **recuperación densa**. La complementariedad *sparse*/denso
  es **débil**: con 92 normas la **colisión de nº de artículo** dispara los falsos positivos
  *wrong-law* de BM25. **Refuta el preliminar del corpus-10** (donde la fusión sí parecía ayudar); es
  un hallazgo honesto de tamaño-de-corpus. Alcance: específico de legislación consolidada consultada en
  lenguaje natural (en jurisprudencia, cf. CLERC, BM25 es competitivo).

## Generación y juez (OE-05/06/07) — cerrado en held-out, con caveats

> Recuperación interna con el perfil ganador **I1_LEGAL**. Generador `qwen2.5:7b-instruct`,
> juez `gemma3:12b` (familia distinta, anti auto-preferencia). Métricas deterministas sin juez;
> **L3 fidelidad / L5 corrección por anotación HUMANA** (el juez se validó y se halló insuficiente).

- **Sistema (*test*, n=28):** responde 17, se abstiene en 11 (**sobre-abstención 39,3 %**); cobertura
  de hechos clave **0,50** [0,32, 0,66] y citas $F_1$ **0,78**. En robustez *dev*+*test* (n=81; 48
  respondidas, 33 con gold) sube a hechos clave **0,58** y citas $F_1$ **0,83**, con IC más estrechos.
- **Seguridad (*out-of-corpus*):** **0/30** respuestas indebidas en dominio lejano (cota superior
  11,6 % por Clopper-Pearson) y **1/10** en *near-miss*; *balanced accuracy* de la decisión
  responder/abstenerse ≈ 0,79. La señal de recuperación separa muy bien la materia ajena (AUC 0,94)
  pero **no** la suficiencia dentro del corpus (near-miss AUC 0,82) → un umbral no recupera utilidad.
- **Descomposición del error (*baselines* en *test*):** *closed-book* rinde 0,23 en hechos clave (sin
  citas) → **recuperar aporta valor real**. El **oráculo** (evidencia *gold*) responde 25/28 y baja la
  sobre-abstención al **10,7 %**, pero **no** mejora el contenido por respuesta (0,44 vs 0,50 del
  sistema en las 17 comunes) → **el cuello del contenido es el generador**; el de la abstención y las
  citas es la **recuperación / ensamblado del contexto** (el oráculo rescata **8 de 11**
  sobre-abstenciones; solo 3 eran fallos de *ranking*). Medir por capas separadas es lo que revela
  esta asimetría.
- **Efecto del tamaño del generador (14B, *test*):** el 14B mejora contenido (hechos clave +0,13 sobre
  las mismas preguntas; mejora **indicativa/borderline** —bootstrap p≈0,03 pero Wilcoxon exacto pareado
  p≈0,07, no confirmada al 5 %) y reduce la sobre-abstención (17,9 %), y **no** confabula de memoria
  (*closed-book*: responde 1/28); pero **rompe la seguridad** (3/30 dominio lejano, **4/10 near-miss**).
  Camino natural: 14B + *gate* de suficiencia. El sistema operativo se mantiene en el **7B** por su
  garantía de seguridad sin *gate*.
- **Frontera utilidad/seguridad (A/B de *prompt*, *dev*):** conservador 33,96 % over-abstención / **0 %**
  respuestas indebidas OOC vs permisivo 9,43 % / **30 %** → **prompt conservador** (seguridad
  innegociable en derecho).
- **Juez (OE-06):** validado contra anotación humana y **rechazado** para fidelidad/corrección —
  sobre-acredita (sensibilidad 2/6 en la clase infiel); AC1 alto por prevalencia, no por acierto en la
  minoría. La referencia humana (*dev*, n=31): **81 %** fieles, **58 %** correctas plenas (IC anchos).
  → resultado **negativo** honesto; L3/L5 se apoyan en anotación humana + señales deterministas.

> **Trazabilidad.** Los identificadores de corrida (*run_id*) y los reports por pregunta se exportan en
> el paquete de evidencia (no en el árbol Git, por tamaño); cada tabla de la memoria es reproducible con
> los comandos del apéndice de documentación.
