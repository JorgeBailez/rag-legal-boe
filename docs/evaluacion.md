# Evaluación

Métricas, dataset y protocolo de evaluación. Documento vivo.

## Estado: Fase 2 (retrieval denso)

Implementadas las **métricas de retrieval denso** (`src/evaluation/metrics.py`) y el **dataset
versionable** `data/evaluation/dense_retrieval_v1/` (`questions.jsonl` + `judgments.jsonl` +
`README.md`). El dataset trae solo ejemplos: la anotación jurídica revisada es posterior y **Gate C**
(`scripts/validate_evaluation_dataset.py`) bloquea el benchmark formal hasta que exista.

- **Métrica primaria**: `ParentnDCG@10`. Controles: `ParentRecall@5`, `EvidenceRecall@5`,
  `ParentHit@1`. Significancia: bootstrap pareado con IC 95 % y semilla fija registrada.
- **Retrieval**: ParentHit@1, ParentRecall@{1,3,5,10}, ParentMRR@10, ParentnDCG@10, EvidenceHit@k,
  EvidenceRecall@k, UniqueParents@k, DuplicateParentRate@k.
- **Contexto**: ContextEvidenceRecall, ContextPrecisionById/RecallById, ContextCharacters,
  ContextItemCount, ExpansionRatio, RedundantContextRate.
- **Rendimiento**: duraciones/throughput/latencias p50/p95, `peak_ram_mb` (cuando disponible).

Protocolo y comandos: `docs/fase2_dense_baseline.md` y `docs/run_dense_embeddings_server.md`. El
dataset legacy `data/evaluation/questions.json` queda para fases posteriores (generación/citas).

## Dimensiones del fallo (a medir por separado)

- **Recuperación**: Recall@k, Precision@k, MRR, "artículo correcto recuperado".
- **Generación**: faithfulness (no contradice ni inventa) y answer relevancy.
- **Citas**: citation accuracy (la cita apunta al bloque que soporta la afirmación).
- **Robustez**: abstención correcta ante preguntas fuera del corpus.

## Tipos de pregunta a cubrir en el dataset

Directa por artículo, ciudadana, conceptual, procedimental, léxica (BM25), comparativa
y sin respuesta (abstención).

## Herramientas

- `src/evaluation/metrics.py` (pendiente de implementación).
- RAGAS: candidato para automatizar parte de la evaluación (no instalado aún).

## Pendientes

Ver `known_issues.md` (estrategia de evaluación y umbral de abstención).
