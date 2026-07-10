# scripts/experiments — experimentos cerrados (reproducibles)

> Ablaciones puntuales ya **ejecutadas y reportadas**. No forman parte del pipeline diario, pero
> se conservan para **reproducir** los hallazgos. La decisión y los números viven en el ledger
> `docs/decisiones_de_diseno.md` (Fase 4) y en `docs/hoja_de_ruta_experimental.md` (experimento E1).

| Script | Ablación | Hallazgo (cerrado) |
|---|---|---|
| `ablate_bm25.py` | OFAT de BM25 (stopwords, stemming, `heading_boost`, k1, b) | el único knob significativo es `heading_boost` (desambigua la colisión de nº de artículo) |
| `ablate_fusion.py` | RRF vs convexa (barrido de α) | convexa con α alto > RRF, pero solo **empata** al denso |
| `ablate_context.py` | ensamblado de contexto L2 (estrategia × k × presupuesto) | se fija `P_EXPAND_BOUNDED · B4K · k=3` |

Se ejecutan desde la raíz del repo, p. ej.:

```bash
uv run python scripts/experiments/ablate_bm25.py --bundle data/indexes/dense/<bundle_id> \
  --split development --gate-c-level checkpoint --threads 24
```

Reutilizan `evaluate_retrieval_strategies` (mismas métricas L1 que el flagship), así que comparten
código vivo con `scripts/benchmark_retrieval_strategies.py`.
