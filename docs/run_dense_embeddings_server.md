# Guía de ejecución — embeddings densos en el servidor universitario

Esta guía permite **generar los índices densos en el servidor** (CPU) sin leer el código. El
servidor objetivo es CPU-only (AMD EPYC 7451, 48 núcleos físicos / 96 lógicos, 125,7 GB RAM, sin
GPU). Todo corre en CPU; por defecto `device=cpu`, `threads=8`.

> El código ya está validado offline con fixtures. Aquí solo se ejecutan las **cargas pesadas**
> (descarga de modelos + codificación) que no caben en el entorno de desarrollo.

## 0. Requisitos previos

- Python 3.11+ y [`uv`](https://docs.astral.sh/uv/). Si falta uv:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- El **corpus procesado de Fase 1** en `data/processed/` (documents, parents, chunks) y el reporte
  `data/processed/reports/mvp_chunking_audit.json` con `pre_embedding_readiness.ready = true`.
  - Si copias el repo, copia también `data/processed/`, `data/manifests/` y `data/raw/boe/`.
  - Si no, regéneralo desde el raw (sin red): `uv run python scripts/process_mvp_corpus.py`.

## 1. Preparar el entorno

```bash
uv sync
uv run pytest -q
```

## 2. Comprobar el corpus (Fase 1)

```bash
uv run python scripts/validate_raw_integrity.py
uv run python scripts/validate_mvp_corpus.py --strict
uv run python scripts/audit_corpus.py --strict   # debe dejar pre_embedding_readiness.ready=true
```

## 3. Ver los modelos disponibles

```bash
uv run python scripts/generate_dense_index.py --list-models
```

Aliases: `e5-base`, `e5-large`, `e5-large-instruct`, `bge-m3`, `qwen3-0.6b`, `gte-multilingual-base`.

## 4. (Opcional) Autenticación de Hugging Face

Solo si un modelo requiere autenticación. **Nunca** guardes el token en el repositorio.

```bash
read -s -p "HF token: " HF_TOKEN
echo
export HF_TOKEN
```

Al terminar, elimina la variable de la sesión:

```bash
unset HF_TOKEN
```

## 5. Fijar las revisiones (reproducibilidad)

El registro (`src/embeddings/model_registry.py`) trae las revisiones **sin fijar**
(`model_revision = None`). Para publicar bundles, fija el commit hash exacto de cada
modelo/tokenizer en ese fichero antes de generar. Para obtener el hash actual de un repo (con red):

```bash
uv run python -c "from huggingface_hub import HfApi; print(HfApi().model_info('BAAI/bge-m3').sha)"
```

`--allow-unpinned-revision` queda limitado a exploración: perfilado de tokenizers, resolución
inicial de hashes y smoke tests. Nunca publica bundles ni sirve para benchmark formal.

## 6. Preflight (sin cargar los pesos del encoder)

```bash
uv run python scripts/generate_dense_index.py --model bge-m3 --preflight-only
```

Valida corpus, contrato, tokenizer, fingerprints, inputs y overflows. No carga pesos completos del
encoder, no codifica y no publica. Puede descargar tokenizer, configuración y metadatos.

## 7. (Opcional) Smoke test de coste por modelo

```bash
uv run python scripts/benchmark_dense_models.py --smoke-test
```

Escribe `data/processed/reports/dense/smoke_tests/<id>/`. Si usas
`--allow-unpinned-revision`, el reporte queda marcado como exploratorio.

## 8. Generación normal (flujo habitual)

```bash
uv run python scripts/generate_dense_index.py --model bge-m3
```

La **barra de progreso** aparece automáticamente. Por defecto: view J1, device cpu, threads 8,
overflow_policy=repair, salida en `data/indexes/dense/`. Al terminar imprime la ruta del bundle y
los comandos exactos de validación y consulta.

## 9. Validar y consultar el bundle

```bash
uv run python scripts/validate_dense_index.py --bundle data/indexes/dense/<bundle_id>

uv run python scripts/query_dense_index.py \
  --bundle data/indexes/dense/<bundle_id> \
  --query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"
```

## 10. Repetir para otro modelo / otra vista

```bash
uv run python scripts/generate_dense_index.py --model e5-large
uv run python scripts/generate_dense_index.py --model qwen3-0.6b
```

Genera **J2** y **C1** solo para los finalistas (las ablaciones):

```bash
uv run python scripts/generate_dense_index.py --model bge-m3 --view J2
uv run python scripts/generate_dense_index.py --model bge-m3 --view C1
```

## 11. Ubicación de los outputs

```
data/indexes/dense/<bundle_id>/{manifest.json, embeddings.npy, rows.jsonl, validation_report.json}
data/processed/reports/dense/smoke_tests/<id>/
data/processed/reports/dense/benchmarks/<id>/
```

`bundle_id = <model_alias>__<view>__<hash12>` combina contrato documental, corpus e inputs
preparados. Es inmutable: no se sobrescribe ni se elimina automáticamente.

## Opciones avanzadas (no necesarias en el flujo normal)

```
--view J1|J2|C1        # vista (default J1)
--threads N            # hilos de CPU (default 8; sweep recomendado: 4, 8, 16)
--batch-size N         # tamaño de batch del encoder (default 32)
--no-progress          # desactiva la barra (CI / logs no interactivos)
--output-root PATH     # raíz de salida (default data/indexes/dense)
--allow-unpinned-revision  # solo preflight exploratorio; no publica
```

Consulta avanzada:

```bash
uv run python scripts/query_dense_index.py \
  --bundle data/indexes/dense/<bundle_id> \
  --query "..." \
  --query-profile-id BASELINE
```

Usa `BASELINE` para modelos cuyo template no contiene `{task}` (`e5-base`, `e5-large`, `bge-m3`,
`gte-multilingual-base`). Los perfiles `I0/I1/I2` solo aplican a modelos instruct; `I_MINUS_NONE`
solo a Qwen3.

Benchmark avanzado:

```bash
uv run python scripts/benchmark_dense_models.py \
  --bundle data/indexes/dense/<bundle_id> \
  --split development \
  --context-ablations
```

Barrido de hilos (medir throughput):

```bash
for t in 4 8 16; do
  uv run python scripts/benchmark_dense_models.py \
    --smoke-test \
    --models bge-m3 \
    --threads "$t"
done
```

## Resolución de problemas

- **Ejecución interrumpida durante la generación**: nada queda publicado (la escritura ocurre en
  `data/indexes/dense/.staging/` y solo se publica con un rename atómico final). Vuelve a lanzar el
  comando. Limpia staging abandonado de forma explícita y separada:
  ```bash
  rm -rf data/indexes/dense/.staging/*
  ```
- **El bundle ya existe**: el comando no sobrescribe. Generar dos veces el mismo contrato/corpus/
  inputs falla correctamente porque los bundles publicados son inmutables; genera un bundle nuevo
  cambiando corpus, inputs, vista o contrato fijado.
- **Error de descarga del modelo / tokenizer**: si el repo requiere autenticación, exporta
  `HF_TOKEN` (paso 4) y repite. No guardes el token en el repositorio.
- **Gate A con errores**: revisa el mensaje (auditoría no lista, revisión sin fijar, overflow…).
  El overflow se repara solo; si aparece "overflow sin reparar" hay un input imposible de dividir
  (presupuesto de tokens no positivo): revisa el modelo elegido.
- **Warnings vs errores**: los WARNING diagnósticos quedan en reportes exploratorios; cualquier
  ERROR detiene la publicación. Las revisiones sin fijar nunca publican.
