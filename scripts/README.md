# scripts — índice de CLIs

Herramientas de línea de comandos del pipeline. Todas se ejecutan **desde la raíz del repo**
(`uv run python scripts/<x>.py …`). El detalle de cada flujo está en el docstring de cada script;
esta tabla es solo el mapa.

## Ingesta y procesado (raw → contratos v2)

| Script | Qué hace |
|---|---|
| `download_boe_raw.py` | Descarga el raw de **una** norma + manifest (con red). |
| `parse_boe_raw.py` | Parsea el raw local de una norma → `documents/histories/parents` v2 (offline). |
| `chunk_boe_document.py` | `document + parents` → `chunks` v2 de una norma (offline). |
| `build_corpus.py` | Descarga + verifica + procesa un **seed** entero (con red). |
| `process_mvp_corpus.py` | Reprocesa **offline** el corpus desde el raw ya descargado (tras cambios de schema). |
| `validate_raw_integrity.py` | sha256/size del raw vs manifests. |
| `validate_mvp_corpus.py` | Valida contratos Pydantic + integridad relacional (offline). |
| `audit_corpus.py` | Auditoría de calidad parser+chunker + *gate* de readiness. |

> **Nota sobre el "mvp" y `audit_corpus`:** su default es `seed_corpus.json` (las **10 normas** del
> prototipo, cuyo raw está **versionado** para reproducibilidad local). Para el **corpus-92** pásales
> `--seed data/corpus/seed_corpus_ampliado.json`; su raw completo no se versiona y debe
> reconstruirse o aportarse localmente. El nombre "mvp" es histórico; los scripts son agnósticos al
> seed.

## Índice denso

| Script | Qué hace |
|---|---|
| `generate_dense_index.py` | Construye un **bundle** denso reproducible (inmutable). |
| `validate_dense_index.py` | Revalida un bundle publicado (Gate B: dtype, dim, norma L2, ids). |
| `query_dense_index.py` | Consulta manual de un bundle (debug/demo). |
| `profile_tokenizers.py` | Perfila tokens de los chunks vs límites del modelo. |

## Evaluación de recuperación (L1)

| Script | Qué hace |
|---|---|
| `build_eval_candidates.py` | Pooling TREC desde varios bundles (material para anotar relevancia). |
| `benchmark_dense_models.py` | Bake-off de modelos densos (selección de modelo, **OE-03**). |
| `benchmark_retrieval_strategies.py` | **Flagship** denso vs BM25 vs híbrido (**OE-04**). |
| `validate_evaluation_dataset.py` | Valida el dataset de evaluación (Gate C). |
| `audit_eval_dataset.py` | Verifica el *grounding* del dataset contra el corpus real. |

## Generación (L3–L6)

| Script | Qué hace |
|---|---|
| `answer_question.py` | CLI de una pregunta → respuesta fundamentada o abstención. |
| `run_generation_eval.py` | Corre el RAG sobre un split + métricas L3–L6 + report versionado. |

## Validación del juez

Los cuatro se parecen en el nombre pero hacen cosas distintas:

| Script | Qué hace | Cuándo |
|---|---|---|
| `validate_judge.py` | **VALIDA** el juez vs anotación humana (κ/AC1). Flujo: `--scaffold` → anotas a mano → `--annotations`. | el paso de validación en sí |
| `rejudge_correctness.py` | Re-juzga **solo corrección (L5)** reusando las respuestas de un report; recalcula además las métricas puras. | cambian referencias/prompt de corrección, generación congelada |
| `rejudge_report.py` | Re-juzga un report entero (**L3+L5**) con un **prompt de juez nuevo** y escribe un report nuevo que consume `validate_judge.py`. | aislar el efecto del prompt, partiendo de un **report** |
| `revalidate_judge_correctness.py` | Re-ejecuta `judge_correctness` (prompt actual) sobre una **anotación ya hecha** y da κ/AC1 + la dirección del cambio. | aislar el efecto del prompt, partiendo de la **anotación** (autocontenido) |

## Dataset (corpus-92) y anotación

| Script | Qué hace |
|---|---|
| `build_corpus92_dataset.py` | Consolida preguntas + judgments verificados → `data/evaluation/corpus92_v1/`. |
| `annotation_worksheet.py` | Convierte el scaffold del juez JSONL ↔ Markdown para anotar a ciegas. |

## Utilidades

| Script | Qué hace |
|---|---|
| `inspect_processed_norm.py` | Vista humana consolidada de una norma (join de los 4 artefactos). |

## experiments/

Experimentos cerrados y reproducibles (ablaciones de BM25, fusión y contexto). Ver
[`experiments/README.md`](experiments/README.md).
