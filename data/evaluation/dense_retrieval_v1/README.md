# Dataset de evaluación — dense_retrieval_v1

Dataset de evaluación del sistema RAG sobre el corpus MVP (10 normas). Versionable y pensado para
**revisión jurídica manual**. Tres ficheros JSONL (una entrada por línea):

- `questions.jsonl` — las preguntas (retrieval + generación).
- `judgments.jsonl` — juicios de relevancia parent↔pregunta (gold de **retrieval**).
- `answer_keys.jsonl` — respuesta de referencia + hechos clave + citas esperadas (gold de
  **generación**, modelo-agnóstico).

> **Estado actual: BORRADOR (`provenance: auto_draft`, `review_status: draft`).** Los enunciados y
> metadatos están autorados según la taxonomía de modos de fallo, y los `parent_id` /
> `expected_citation_parents` reutilizan mapeos **verificados del corpus**, pero **nada está
> `reviewed`**: faltan la revisión jurídica, los `paragraph_orders` reales, la relevancia graduada
> (1/0) y el ajuste de `reference_answer`/`key_facts`. Hasta esa revisión, **Gate C no habilita** el
> benchmark formal. Diseño y métricas: [`docs/evaluacion_gold_y_metricas.md`](../../../docs/evaluacion_gold_y_metricas.md).

## `questions.jsonl`

| campo | obligatorio | valores | nota |
|---|---|---|---|
| `query_id` | sí | p. ej. `q0001` | único |
| `query` | sí | texto | pregunta en español |
| `split` | sí | `development` \| `test` \| `out_of_corpus` | una `issue_family_id` no puede estar en development **y** test |
| `issue_family_id` | sí | p. ej. `plazo_resolver_solicitud` | agrupa variantes de un mismo asunto (evita fuga dev/test) |
| `query_style` | sí | `directa_articulo` \| `ciudadana` \| `conceptual` \| `procedimental` \| `lexica` \| `comparativa` \| `sin_respuesta` | vocabulario recomendado (fuera de lista → aviso) |
| `answer_scope` | sí | `single_parent` \| `multi_parent` \| `none` | `none` solo en `out_of_corpus` |
| `difficulty` | no | `facil` \| `media` \| `dificil` | para estratificar resultados |
| `failure_mode` | no | p. ej. `numeric_threshold`, `false_premise`, `cross_norm`, `temporal_without_content`, `lexical_term`, `in_corpus_unanswerable`, `out_of_corpus` | modo de fallo que la pregunta busca provocar |
| `provenance` | no | `auto_draft` \| `human_authored` | procedencia real |
| `review_status` | sí | `example` \| `draft` \| `reviewed` \| `final` | solo `reviewed`/`final` cuentan para Gate C |
| `notes` | no | texto | observaciones |

## `judgments.jsonl`

| campo | obligatorio | valores | nota |
|---|---|---|---|
| `query_id` | sí | referencia a una pregunta | debe existir |
| `parent_id` | sí | `BOE-A-...__<block>` | bloque jurídico juzgado; debe existir en el corpus |
| `relevance` | sí | `2` \| `1` \| `0` | 2 = central/suficiente · 1 = apoyo/matiz · 0 = revisado y descartado · *ausencia* = no juzgado |
| `evidence` | no | `{ "paragraph_orders": [..] }` | párrafos del parent que sustentan la relevancia |
| `quote` | no | texto | cita literal opcional |
| `review_status` | sí | `example` \| `draft` \| `reviewed` \| `final` | |
| `notes` | no | texto | |

## `answer_keys.jsonl`

| campo | obligatorio | valores | nota |
|---|---|---|---|
| `query_id` | sí | referencia a una pregunta | uno por pregunta |
| `answerable` | sí | `true` \| `false` | `false` ⇒ gold de abstención (OOC e in-corpus-sin-respuesta) |
| `reference_answer` | no | texto | respuesta canónica breve (obligatoria si `reviewed` y `answerable`) |
| `key_facts` | no | lista de strings | hechos que una respuesta correcta debe contener (p. ej. `"un mes"`, `"40.000"`) |
| `forbidden_facts` | no | lista de strings | trampas (p. ej. `"tres meses"`); su presencia señala alucinación |
| `expected_citation_parents` | no | lista de `parent_id` | citas correctas (⊆ corpus; obligatoria si `reviewed` y `answerable`) |
| `review_status` | sí | `example` \| `draft` \| `reviewed` \| `final` | |
| `notes` | no | texto | |

## Reglas (validador / Gate C)

```
uv run python scripts/validate_evaluation_dataset.py                       # informe
uv run python scripts/validate_evaluation_dataset.py --strict              # exit≠0 si errores estructurales
uv run python scripts/validate_evaluation_dataset.py --require-gate-c      # exit≠0 si Gate C no listo
uv run python scripts/validate_evaluation_dataset.py --gate-c-level checkpoint
```

- **Estructurales (siempre):** contrato válido, `query_id` único, juicio/answer_key sin pregunta,
  `(query_id, parent_id)` duplicado, `parent_id`/`expected_citation_parents` inexistentes en el
  corpus, `paragraph_orders` inexistentes, `relevance` fuera de `{0,1,2}`, juicio relevante en
  `out_of_corpus`, `answer_scope='none'` ↔ `out_of_corpus`, `answer_key` `answerable` en
  `out_of_corpus`, y **fuga de `issue_family_id` entre development y test**.
- **Completitud (solo `reviewed`/`final`):** evidencia para `relevance=2`, justificación para
  `relevance≥1`, `multi_parent` con ≥2 parents relevantes, `reference_answer` y
  `expected_citation_parents` no vacíos para answerable. *Los borradores pueden estar incompletos.*
- **Gate C retrieval:** mínimos de preguntas revisadas con juicio relevante por split.
- **Gate C generación:** además, answer_keys revisados que cubren esas preguntas.
- **Niveles:** checkpoint = 40 dev / 20 test / 10 OOC · formal = 40 dev / 80 test / 20 OOC.

## Cómo anotar (silver → gold)

1. Revisa el enunciado (`provenance: auto_draft` → `human_authored` si lo reescribes).
2. Recupera candidatos con `scripts/query_dense_index.py` y, para no sesgar, también con otros
   recuperadores cuando existan (pooling multi-sistema).
3. En `judgments.jsonl`: ajusta la relevancia (2/1/0), marca los `paragraph_orders` reales (con
   `scripts/inspect_processed_norm.py`) y la `quote`.
4. En `answer_keys.jsonl`: revisa `reference_answer`, `key_facts`, `forbidden_facts` y
   `expected_citation_parents`.
5. Marca `review_status: reviewed` **solo** cuando esté verificado jurídicamente, y valida con el
   comando de arriba antes del benchmark formal.

## Cobertura (modos de fallo)

Directa por artículo, ciudadana/paráfrasis, conceptual, procedimental, **léxica** (BDNS, ICIO,
DPO), **numérica** (40.000/15.000 €, 5 %, 500.000 €, plazos, 4 años), **comparativa/cross-norm**
(alzada vs reposición; Consejo de Transparencia vs alzada), **desambiguación**, **premisa falsa**,
**trampa temporal** (artículo «(Sin contenido)»), **in-corpus sin respuesta** y **out-of-corpus**.

## Resultados de checkpoint (Fase 2, retrieval)

- [`checkpoint_baseline_dense.md`](checkpoint_baseline_dense.md) — selección provisional del
  baseline denso y ablación de vistas documentales.
