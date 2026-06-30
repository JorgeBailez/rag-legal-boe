# RESULTADOS — síntesis de evidencia (corpus-92)

> Resultados **vigentes** del TFG sobre el **corpus-92** (gold `corpus92_v1`, de-sesgado por
> re-pooling con BM25+híbrido). La evidencia completa —números, IC y cómo reproducir— vive en el
> ledger `docs/decisiones_de_diseno.md` (Fases 4 y 5); el estado vivo en `PROGRESO.md`.
> El registro histórico del MVP de 10 normas (incluido lo marcado VOID) está archivado en
> `docs/archive/RESULTADOS_mvp_corpus10.md`.

---

## Recuperación (OE-02/03/04) — N1, cerrada

- **Modelo denso (bake-off-92, dev n=53, `bench_20260624T181000Z`):** ganador
  **`e5-large-instruct · J1 · I1_LEGAL`** (ParentnDCG@10 **0.802**) > bge-m3 (0.719) > e5-base (0.627),
  **significativo en pareado** y **Pareto-óptimo** (bge-m3 dominado). La **instrucción de query es
  palanca** (I1_LEGAL +0.070 SIG sobre I0_GENERIC). → **OE-03 cerrada.**
- **Flagship denso vs BM25 vs híbrido (test held-out n=28, `retrieval_20260625T111234Z`):**
  **denso 0.797 GANA**; convexa α0.7 0.769 (Δ−0.028 **n.s.**); RRF 0.706 (Δ−0.091 **SIG peor**);
  BM25 0.507 (SIG peor). Ablaciones en dev: BM25 → **`heading_boost`** es el único knob significativo;
  fusión → convexa α alto > RRF pero solo **empata** al denso. → **OE-04 cerrada.**
- **Tesis OE-04:** con un embedder instruct fuerte sobre legislación consolidada del BOE, el **denso es
  robustamente superior** y la **fusión no mejora** (el RRF empeora). La complementariedad sparse/denso
  es **débil**: con 92 normas la **colisión de nº de artículo** dispara los falsos positivos wrong-law
  de BM25 (re-pooling: 127/138 candidatos BM25 = rel=0). **Refuta el preliminar del corpus-10** (que
  daba `directa_articulo` 0.41→0.75); es un hallazgo honesto de tamaño-de-corpus. **→ El sistema usa
  recuperación DENSA.**
- Se mantiene del MVP: la abstención por umbral de score funciona (en 92, e5-large/I1 AUC **0.97**);
  e5-large-instruct = mejor calidad/coste, bge-m3 dominado.

## Generación y juez (OE-05/06/07) — cerrado en held-out, con caveats

> Detalle y trazabilidad en `PROGRESO.md` y `docs/analisis_errores_generacion.md`. El juez LLM se
> validó y se halló **insuficiente** (sobre-acredita; L3 sensibilidad 2/6), así que **L3 fidelidad /
> L5 corrección se reportan por anotación HUMANA**, no por el juez (κ/AC1 v2+v3 documentan la
> validación). Generador `qwen2.5:7b-instruct`, juez `gemma3:12b` (familia distinta).

- **Generación v1 (segura), corrida final en `test` (`gen_20260630T091038Z`) + OOC (`…091513Z`):**
  **0 respuestas falsas en OOC** (30/30 abstención correcta), **0 hechos prohibidos**; over-abstención
  **35.7%** (answer_rate 0.64); calidad (n=10) key-fact 0.72 (suelo léxico), citas F1/recall 0.80.
  Auditoría manual de las 18 respondidas: **0 alucinaciones de dato**.
- **Frontera utilidad/seguridad (A/B de prompt):** v1 (segura) 0% false-answer / 36% over-abstención
  vs v2 (útil) 30% false-answer / 9% over-abstención → **v1 operativa** (seguridad innegociable). El
  **gate de abstención por score** (piloto L6 AUC 0.97) queda como trabajo futuro documentado.
- **Taxonomía de errores (OE-07):** dos patrones — (1) colisión de recuperación → artículo equivocado
  (lo caza el score-gate); (2) sobre-extensión del 7B → detalle no soportado (no lo caza ningún gate).
