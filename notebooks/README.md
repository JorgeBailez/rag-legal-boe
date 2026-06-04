# Notebooks — RAG Legal BOE

Los notebooks son **consumidores de reportes**: leen los artefactos generados por los scripts
(`data/processed/reports/dense/...`) y construyen la narrativa y las figuras del TFG. **No** generan
embeddings ni publican bundles (eso lo hacen los scripts, normalmente en el servidor universitario).

| Notebook | Contenido |
|---|---|
| `01_exploracion_api_boe.ipynb` | Cómo se construyó el corpus jurídico (API → raw → modelo documental → parser → chunking). |
| `02_perfilado_tokenizacion.ipynb` | Límites reales de los tokenizadores; por qué se prohíbe el truncamiento silencioso. |
| `03_benchmark_modelos_densos.ipynb` | Calidad frente a coste CPU: smoke tests, latencia, RAM, throughput, instrucciones. |
| `04_ablaciones_chunking_y_contexto.ipynb` | J1 vs J2 vs C1; K_ONLY vs P_EXPAND_*; B4K/B8K/B12K; barrido de k. |
| `05_seleccion_baseline_dense.ipynb` | Síntesis final, tabla comparativa, figuras, decisión y limitaciones. |

## Cómo ejecutarlos

```bash
uv sync
uv run jupyter notebook notebooks/<nombre>.ipynb
```

- Deben funcionar con **Restart Kernel + Run All**.
- **Fallan de forma clara** si faltan los reportes que consumen (primero hay que ejecutar los
  scripts correspondientes, normalmente en el servidor; ver `docs/run_dense_embeddings_server.md`).
- Usan **rutas relativas** desde la raíz del repo y muestran `benchmark_id` y fingerprints para
  trazabilidad.
- Narrativa en **español**; código y nombres técnicos en **inglés**.

## Salidas

```
data/processed/reports/dense/notebooks/   # tablas/datos derivados de los notebooks
data/processed/reports/dense/figures/     # figuras generadas
thesis/figures/                           # figuras finales seleccionadas para la memoria
```

## Reportes que consume cada notebook

- `02`: `data/processed/reports/tokenizer_profile.json` (de `scripts/profile_tokenizers.py`).
- `03`: `data/processed/reports/dense/smoke_tests/<id>/` (de `scripts/benchmark_dense_models.py --smoke-test`).
- `04` y `05`: `data/processed/reports/dense/benchmarks/<id>/` (de `scripts/benchmark_dense_models.py`).
