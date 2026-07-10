# Fase 2 — Baseline denso (dense-only)

> Nota de alcance. Este documento describe el diseño del índice denso, que sigue vigente. Las cifras
> y los bundles que cita son de la primera fase (corpus de 10 normas) y han quedado superados por el
> corpus de 92: el modelo elegido es `e5-large-instruct · J1 · I1_LEGAL` y el recuperador del sistema
> es el denso. La comparación con BM25 y la fusión híbrida llegó en una fase posterior y confirmó que,
> en este corpus, el denso es la mejor opción. Las decisiones con su evidencia están en
> `decisiones_de_diseno.md`.

Documentación de la Fase 2: embeddings densos reproducibles, índice exacto, consulta, ensamblado de
contexto y evaluación. Documento vivo.

## Objetivo

Construir, de forma **trazable y reproducible**, un baseline de recuperación **dense-only** sobre el
corpus MVP (10 normas) y dejar el flujo preparado para ejecutar las cargas pesadas en una máquina
con recursos suficientes
CPU. El valor es la trazabilidad y la evaluación, no la escala.

## Fuera de alcance

BM25, retrieval híbrido, sparse productivo, reranking, generación con LLM, chatbot, frontend, API
final, ANN, Chroma/Qdrant, Docker, MLflow/Papermill, ONNX/OpenVINO/FlagEmbedding. No se añaden
abstracciones para backends inexistentes.

## Hardware objetivo

CPU AMD EPYC 7451 (48 físicos / 96 lógicos), 125,7 GB RAM, **sin GPU**. Defaults: `device=cpu`,
`threads=8` (sweep posterior 4/8/16), `processes=1`. No se usan todos los hilos por defecto.

## Texto vectorizado y vistas

La preparación de inputs (`src/embeddings/input_preparation.py`) genera *rows* derivadas por vista:

- **J1** (baseline): `chunks[].retrieval_text` (con contexto jurídico).
- **J2** (ablación): `chunks[].text` (texto crudo del child).
- **C1** (chunking clásico controlado): ventanas fijas token-aware dentro de cada parent
  (`parents[].text`), overlap 100 tokens, **sin cruzar parents**.

El `context_anchor` (rango de párrafos del parent) se resuelve para J1/J2 buscando el chunk como una
secuencia contigua exacta de párrafos. En C1 se calcula por solape token-aware de cada ventana con
los párrafos del parent, y en `overflow_repair` se conserva/refina el anchor del chunk origen.

## Overflow repair y prohibición de truncado

Antes de codificar: texto base → `format_document` del contrato → tokenizador real (con special
tokens) → conteo → validación. Si un input excede el límite efectivo se **repara** dividiéndolo en
ventanas token-aware (overlap 100), conservando trazabilidad (`origin="overflow_repair"`,
`segment_index/segment_count`, `token_start/token_end`). **Truncamiento silencioso prohibido**: si
queda algún overflow sin reparar, la generación se detiene (error bloqueante) y no se publica nada.

## Registro de modelos y contratos

`src/embeddings/model_registry.py` declara, por modelo: `alias`, `model_id`,
`model_revision`/`tokenizer_revision` (commit hash exacto), `declared_max_tokens`,
`expected_embedding_dimension`, `document_template`, `pooling`, `normalize_embeddings`,
`default_query_template`/`default_query_instruction`, `trust_remote_code`/`remote_code_reviewed`,
`notes`. Se separan conceptualmente:

- **document_embedding_contract**: lo que define la identidad de los embeddings documentales.
- **query_profile**: cómo se formatea la query (configurable, no altera los embeddings de documento).

Shortlist: `e5-base`, `e5-large`, `e5-large-instruct`, `bge-m3`, `qwen3-0.6b`,
`gte-multilingual-base`. `trust_remote_code=False` por defecto; explícito solo donde el modelo lo
exige (gte). **Revisiones sin fijar** por defecto: el gate bloquea la publicación hasta fijarlas.
`--allow-unpinned-revision` solo se admite para exploración (tokenizer profiling, resolución inicial
de hashes y smoke tests), nunca para publicar ni para benchmark formal.

## Encoder

`src/embeddings/encoder.py` → un único `DenseEncoder` sobre Sentence Transformers: CPU explícito,
float32, normalización L2, dimensión validada, `batch_size`/`progress` configurables. Los documentos
llegan ya formateados por la preparación (se codifican verbatim). Las queries se formatean con un
`query_profile_id` reproducible (`I0_GENERIC`, `I1_LEGAL`, `I2_CITIZEN_LEGISLATION`;
`I_MINUS_NONE` solo para Qwen3). Cambiar el perfil de query no regenera embeddings documentales.
`HF_TOKEN` se lee solo del entorno y nunca se imprime/persiste/incluye en excepciones. Barra de
progreso visible por defecto (`--no-progress` para CI).

## Bundle y fingerprints

`src/embeddings/bundle.py`. Bundle inmutable en
`data/indexes/dense/<bundle_id>/{manifest.json, embeddings.npy, rows.jsonl, validation_report.json}`,
con `bundle_id = <model_alias>__<view_lower>__<bundle_identity_hash_12>`, donde la identidad combina
contrato documental, fingerprint del corpus e inputs preparados. `embeddings.npy` es float32
`[n_rows, dim]`; la carga pública es `load_validated_bundle(...)`, que valida schema, checksums,
Gate B, revisiones fijadas, corpus actual y parents existentes. El manifest es legible y anidado
(`schema_version, bundle, corpus, document_embedding_contract, execution, artifacts, validation`).
Fingerprints distintos sobre JSON canónico (SHA-256):
`source_corpus_fingerprint`, `embedding_inputs_fingerprint`, `document_contract_fingerprint`.
Publicación segura: staging → validar → checksums → manifest → **rename atómico**. Si algo falla, no
se publica y se limpia el staging. Un bundle publicado nunca se sobrescribe.

## Gates

- **Gate A** (pre-encoding): auditoría aprobada (`pre_embedding_readiness.ready`), modelo registrado,
  revisiones fijadas para publicación, código remoto revisado si aplica, tokenizer cargable, inputs
  preparados, ids únicos, anchors presentes/válidos, overflow resuelto, truncamiento = 0.
- **Gate B** (pre-publicación): existen los artefactos; `n_rows` == filas; shape/dtype float32;
  NaN/Inf/vectores nulos = 0; norma L2 ≈ 1; `row_index` continuo; `embedding_input_id` único;
  `parent_id` presente; checksums y fingerprints correctos.
- Severidades: ERROR bloquea; WARNING publica pero queda en el reporte; INFO diagnóstico. Dos ids
  idénticos = ERROR; dos inputs distintos con vector idéntico = WARNING.

## Índice exacto y filtros

`src/indexing/vector_index.py` → `ExactDenseIndex` (sin interfaz abstracta): rows en memoria,
`embeddings.npy` por mmap de solo lectura; búsqueda exacta por producto escalar
(`embeddings @ q`, `argsort` estable). Filtros opcionales por máscara en memoria (join ligero a
chunks/descriptors): `rank_code, scope_code, subject_codes, semantic_role, without_content, annex,
table, image`. El benchmark principal corre sin filtros.

## Context assembler

`src/retrieval/context_assembler.py` (separado de retrieval y generación): `K_ONLY`,
`P_EXPAND_FULL`, `P_EXPAND_BOUNDED`. Bounded expande alrededor del anchor alternando posterior/
anterior, manteniendo el orden jurídico y sin superar el presupuesto. Presupuestos evaluables:
B4K=4000, B8K=8000 (candidato provisional), B12K=12000 caracteres.

## Dataset de evaluación

`data/evaluation/dense_retrieval_v1/` (`questions.jsonl` + `judgments.jsonl` + `README.md`).
Relevancia 2/1/0 (ausencia = no juzgado). Splits `development/test/out_of_corpus`; una
`issue_family_id` no puede estar en development y test. **Gate C** bloquea los benchmarks formales
si no hay anotación revisada suficiente. Niveles: `checkpoint` (40 development, 20 test, 10
out_of_corpus) y `formal` (40 development, 80 test, 20 out_of_corpus). El validador comprueba
parents/párrafos contra el corpus, duplicados, evidencia y reglas de out-of-corpus/multi-parent. El
scaffold trae solo ejemplos (revisión jurídica humana posterior). Validador:
`scripts/validate_evaluation_dataset.py`.

## Métricas

`src/evaluation/metrics.py`. Retrieval: `ParentHit@1`, `ParentRecall@{1,3,5,10}`, `ParentMRR@10`,
`ParentnDCG@10` (**primaria**), `EvidenceHit@k`, `EvidenceRecall@k`, `UniqueParents@k`,
`DuplicateParentRate@k`. Contexto: `ContextEvidenceRecall`, `ContextPrecisionById`,
`ContextRecallById`, `ContextCharacters`, `ContextItemCount`, `ExpansionRatio`,
`RedundantContextRate`. Rendimiento: duraciones/throughput/latencias p50/p95, `peak_ram_mb` (cuando
está disponible), tamaños. k de retrieval {1,3,5,10}; de contexto {1,3,5,8,10}. **Bootstrap pareado**
con IC 95 % y semilla fija registrada. Controles: ParentRecall@5, EvidenceRecall@5, ParentHit@1.

## Reportes

Regenerables, fuera de Git:
`data/processed/reports/dense/smoke_tests/<id>/{report.json, models.csv}` y
`.../benchmarks/<id>/{report.json, metrics.csv, query_results.jsonl, context_results.jsonl}`.

## Notebooks

`notebooks/02–05` consumen reportes (no generan embeddings). Ver `notebooks/README.md`.

## Checkpoints

P2.1 dependencias/settings/registro · P2.2 contratos/schemas · P2.3 preparación de inputs ·
P2.4 encoder · P2.5 bundle/validación · P2.6 índice/consulta · P2.7 dataset/Gate C ·
P2.8 métricas/assembler · P2.9 reportes/notebooks · P2.10 scripts/docs/QA.

## Deuda futura

- Fijar los commit hashes de los modelos antes de publicar bundles definitivos.
- Anotación jurídica del dataset (Gate C) para el benchmark formal.
- `peak_ram_mb` solo en Linux; en otros SO queda `null`.
- Capas posteriores fuera de esta fase: BM25, híbrido, reranking, generación con LLM, API.
