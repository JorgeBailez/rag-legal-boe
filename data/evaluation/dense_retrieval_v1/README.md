# Dataset de evaluación — dense_retrieval_v1

Dataset de evaluación del **retrieval denso** sobre el corpus MVP (10 normas). Es versionable y
está pensado para **revisión jurídica manual**. Dos ficheros JSONL (una entrada por línea):

- `questions.jsonl` — las preguntas.
- `judgments.jsonl` — los juicios de relevancia parent↔pregunta.

> El scaffold incluye solo **ejemplos** (`review_status: "example"`) para ilustrar el formato. Esos
> ejemplos **no** habilitan el benchmark formal: Gate C exige mínimos revisados y consistencia con
> el corpus procesado.

## `questions.jsonl`

| campo | obligatorio | valores | nota |
|---|---|---|---|
| `query_id` | sí | p. ej. `q0001` | único |
| `query` | sí | texto | pregunta en español |
| `split` | sí | `development` \| `test` \| `out_of_corpus` | una `issue_family_id` no puede estar en development **y** test |
| `issue_family_id` | sí | p. ej. `plazos_resolucion` | agrupa variantes de un mismo asunto (evita fuga dev/test) |
| `query_style` | sí | `directa_articulo` \| `ciudadana` \| `conceptual` \| `procedimental` \| `lexica` \| `comparativa` \| `sin_respuesta` | vocabulario recomendado |
| `answer_scope` | sí | `single_parent` \| `multi_parent` \| `none` | alcance esperado de la respuesta |
| `review_status` | sí | `example` \| `draft` \| `reviewed` \| `final` | el benchmark formal usa `reviewed`/`final` |
| `notes` | no | texto | observaciones del anotador |

## `judgments.jsonl`

| campo | obligatorio | valores | nota |
|---|---|---|---|
| `query_id` | sí | referencia a una pregunta | debe existir en `questions.jsonl` |
| `parent_id` | sí | `BOE-A-...__<block>` | bloque jurídico juzgado |
| `relevance` | sí | `2` \| `1` \| `0` | 2 = central/suficiente · 1 = apoyo/matiz · 0 = revisado y descartado · *ausencia* = no juzgado |
| `evidence` | no | `{ "paragraph_orders": [..] }` | párrafos del parent que sustentan la relevancia |
| `quote` | no | texto | cita literal opcional |
| `review_status` | sí | `example` \| `draft` \| `reviewed` \| `final` | |
| `notes` | no | texto | |

## Reglas (validador / Gate C)

```
uv run python scripts/validate_evaluation_dataset.py            # informe
uv run python scripts/validate_evaluation_dataset.py --strict   # exit≠0 si hay errores estructurales
uv run python scripts/validate_evaluation_dataset.py --require-gate-c  # exit≠0 si Gate C no está listo
uv run python scripts/validate_evaluation_dataset.py --gate-c-level checkpoint
```

- **Errores** (bloquean): contrato inválido, `query_id` duplicado, juicio sin pregunta, `relevance`
  fuera de `{0,1,2}`, `(query_id, parent_id)` duplicado, `parent_id` inexistente,
  `paragraph_orders` inexistentes, evidencia ausente para `relevance=2`, falta de justificación en
  `relevance=1/2`, juicio relevante en `out_of_corpus`, `multi_parent` con menos de dos parents
  relevantes y **fuga de `issue_family_id` entre development y test**.
- **Gate C checkpoint**: 40 preguntas development, 20 test y 10 out_of_corpus revisadas/listas.
- **Gate C formal**: 40 preguntas development, 80 test y 20 out_of_corpus revisadas/listas.

## Cómo anotar

1. Añade preguntas a `questions.jsonl` (empieza con `review_status: "draft"`).
2. Recupera candidatos con `scripts/query_dense_index.py` sobre un bundle ya generado.
3. Juzga cada parent relevante en `judgments.jsonl` (2/1/0) con su `evidence`.
4. Marca `review_status: "reviewed"` cuando el juicio esté revisado jurídicamente.
5. Valida con el comando de arriba antes de lanzar el benchmark formal.
