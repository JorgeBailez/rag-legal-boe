# RAG Legal BOE (TFG)

Sistema RAG para consulta **informativa** de legislación consolidada del BOE, orientado a
ciudadanos no expertos. **No** es asesoramiento jurídico vinculante: los textos
consolidados del BOE tienen carácter informativo y no valor jurídico oficial.

> **Estado (2026-06-25): pata de RECUPERACIÓN cerrada con evidencia (OE-01..OE-04).**
> Flujo extremo a extremo sobre **92 normas**: raw XML inmutable → manifest → parser →
> `documents + histories + parents` → chunks vector-ready → auditoría con *gate* →
> **embeddings densos reproducibles → bundle inmutable → índice exacto (numpy + mmap) →
> recuperación evaluada (denso vs BM25 vs híbrido)** → **generación fundamentada con LLM local
> (Ollama) → JSON estructurado validado con citas oficiales o abstención**.
>
> **Recuperador del sistema: denso `e5-large-instruct · vista J1 · I1_LEGAL`** (bake-off-92, OE-03).
> El **flagship** denso vs BM25 vs híbrido (OE-04) está **cerrado: el denso gana**; la fusión con
> BM25 no mejora en este corpus (evidencia y números en `docs/decisiones_de_diseno.md`). Núcleo
> restante: **generación + validación del juez** (OE-05/06/07). Falta la API (FastAPI), fase
> posterior. Los bundles densos y los pesos **no se versionan** (`.gitignore`): se generan en GPU/CPU.
>
> La **vigencia** de cada bloque se decide por la `fecha_actualizacion` de `indice.xml`
> (coincidencia exacta y única con una versión, que además es la de fecha máxima); el orden de
> las `<version>` en el XML **no** es criterio. Los bloques que no resuelven van a *cuarentena*
> (no indexables, sin chunks) y bloquean el avance a embeddings.

## Requisitos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.

### Instalar uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Puesta en marcha

```bash
# 1. Sincronizar dependencias (crea el entorno .venv)
uv sync

# 2. Crear el fichero de configuración local
#    (copiar .env.example a .env y ajustar si hace falta)
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env

# 3. Ejecutar los tests
uv run pytest

# 4. Lint y formato
uv run ruff check .
uv run ruff format .
```

No se necesitan secretos para arrancar: `Settings()` funciona con valores por defecto.

## Estructura

```
src/        Código fuente (boe, preprocessing, indexing, retrieval,
            generation, evaluation, app, core, config)
data/       Datos (raw, processed, evaluation, manifests) — contenido no versionado;
            sí se versionan el catálogo semilla, los manifests y los reportes de verificación
notebooks/  Exploración (notebook narrativo del recorrido del pipeline)
prompts/    Plantillas de prompt
tests/      Suite de pytest
thesis/     Memoria del TFG (documento de la memoria; seguimiento del tutor)
```

## Descarga raw BOE

Descarga la respuesta raw de una norma consolidada (sin parsear) y genera un manifest
reproducible con hashes y tamaños:

```bash
uv run python scripts/download_boe_raw.py BOE-A-2015-10565
```

Salida:

```
data/raw/boe/BOE-A-2015-10565/     # full.xml, metadatos.xml, analisis.xml,
                                   # metadata_eli.xml, texto.xml, indice.xml
data/manifests/BOE-A-2015-10565.json
```

El contenido XML descargado **no se versiona** (ver `.gitignore`); el manifest sí, como
evidencia de reproducibilidad. La base de la API se configura con `BOE_API_BASE`
(ver `.env.example`). Llama a la API externa del BOE.

## Parseo BOE (raw → artefactos v2)

Convierte el raw local de una norma ya descargada en una representación intermedia neutral y
deriva los **tres artefactos** persistidos. No llama a internet:

```bash
uv run python scripts/parse_boe_raw.py BOE-A-2015-10565
```

Lee `data/raw/boe/<norm_id>/{metadatos,analisis,indice,texto}.xml` +
`data/manifests/<norm_id>.json` y escribe `data/processed/documents/<id>.json`
(descriptor `boe_legal_document_v2`), `data/processed/histories/<id>.json`
(`boe_legal_history_v2`) y `data/processed/parents/<id>.json` (`boe_legal_parents_v2`,
propietario único del texto vigente). No usa `full.xml` como fuente ni parsea `metadata_eli.xml`.

## Contratos de datos v2

La representación procesada **autoritativa es compuesta**: `documents` (descriptor legible) +
`histories` (versiones, notas de modificación, resolución temporal) + `parents` (texto vigente y
párrafos, una sola vez). `chunks` es una **proyección vector-ready mínima** y `reports` una
proyección de auditoría. Modelos Pydantic en `src/contracts/` (fuente única) → JSON Schema en
`schemas/`, validados en runtime al persistir y por `validate_mvp_corpus.py`.
Vista humana por joins: `uv run python scripts/inspect_processed_norm.py <id> [--block <bid>]`.

## Chunking jurídico

Convierte el descriptor + parents en chunks recuperables `boe_legal_chunks_v2`
(parent-child: el bloque jurídico es el padre). Chunking por párrafos, sin red:

```bash
uv run python scripts/chunk_boe_document.py BOE-A-2015-10565
```

Lee `documents/<id>.json` + `parents/<id>.json` y escribe `chunks/<id>.json` (no se versiona).
Solo genera chunks de bloques **indexables**; los bloques resueltos no indexables (encabezados
con texto, firmas, notas iniciales) conservan **parent** pero no generan chunks. Cada chunk
lleva `parent_id`, `citation`, `filters` mínimos y `retrieval_text` con contexto jurídico —
**sin** `parent_text` ni metadatos documentales (se resuelven por join). Aún no hace embeddings.

## Corpus (92 normas)

El corpus actual son **92 normas** en `data/corpus/seed_corpus_ampliado.json` (el `seed_corpus.json`
de 10 normas fue el MVP, histórico). Este comando descarga, **verifica** (vigente +
`estado_consolidacion = Finalizado` + endpoints obligatorios) y procesa (parser + chunking) las
normas que cumplen criterios (con `--seed data/corpus/seed_corpus_ampliado.json`). Llama a la API del BOE:

```bash
uv run python scripts/build_corpus.py
```

Escribe `data/corpus/verification_report.json` (versionado, como evidencia) e imprime una
tabla. Las normas que **no** cumplen criterios se excluyen del procesado y se reportan —
**no se sustituyen automáticamente**.

## Auditoría de calidad del corpus

Audita (solo lectura) que parser y chunker generan lo esperado sobre el corpus
(integridad, trazabilidad XML→documento→chunk, `retrieval_text`, overlap, oversized,
jerarquía, eficiencia). Escribe `data/processed/reports/mvp_chunking_audit.{json,csv}`:

```bash
uv run python scripts/audit_corpus.py
```

Reproceso local del corpus (parser + chunker desde el raw, **sin red**) y validación:

```bash
uv run python scripts/process_mvp_corpus.py        # regenera documents/histories/parents/chunks
uv run python scripts/validate_raw_integrity.py    # sha256/size del raw vs manifests
uv run python scripts/validate_mvp_corpus.py --strict  # contratos Pydantic + auditoría relacional
uv run python scripts/audit_corpus.py --strict     # exit≠0 si readiness no está lista
```

El *gate* `pre_embedding_readiness.ready` solo es `true` si: 0 errores estructurales, 0
divergencias temporales (mismatches/cuarentena/chunks no vigentes), `raw_integrity.ready`
verde, sin fugas de notas y sin contenido sustantivo fuera de retrieval. Único diferido:
`H3_oversized_token_measurement` (se cierra midiendo tokens en la fase de indexación).

## Notebook de exploración

`notebooks/01_exploracion_api_boe.ipynb` documenta el recorrido y las decisiones (API → raw →
modelo documental → parser → chunking → corpus). Requiere las dependencias dev
(`uv sync`) y se ejecuta a mano:

```bash
uv run jupyter notebook notebooks/01_exploracion_api_boe.ipynb
```

## Fase 2 — Índice denso (dense-only)

Embeddings densos reproducibles + índice exacto + consulta + evaluación, partiendo de
`data/processed/chunks/<norm_id>.json`. El índice es **denso** (numpy + mmap). La comparación con
BM25/híbrido (flagship OE-04) se ejecuta aparte con `scripts/benchmark_retrieval_strategies.py` y
**concluyó que el denso gana** (la fusión no mejora; ver `docs/decisiones_de_diseno.md`). El reranking
queda como trabajo futuro. El modelo de embeddings **no** es un default global: se elige con `--model`.

```bash
uv run python scripts/generate_dense_index.py --list-models        # aliases disponibles
uv run python scripts/generate_dense_index.py --model bge-m3 --preflight-only
uv run python scripts/generate_dense_index.py --model bge-m3       # genera el bundle (barra de progreso)
uv run python scripts/validate_dense_index.py --bundle data/indexes/dense/<bundle_id>
uv run python scripts/query_dense_index.py --bundle data/indexes/dense/<bundle_id> \
  --query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"
```

Las **cargas pesadas** (descarga de modelos + codificación) se ejecutan en un servidor CPU; el
código se valida offline con fixtures. Los bundles publicados son inmutables y requieren revisiones
exactas de modelo/tokenizer; `--allow-unpinned-revision` queda solo para exploración sin
publicación. Guía operativa completa:
[`docs/run_dense_embeddings_server.md`](docs/run_dense_embeddings_server.md). Diseño y decisiones:
[`docs/fase2_dense_baseline.md`](docs/fase2_dense_baseline.md). Autenticación opcional de Hugging
Face mediante la variable de entorno `HF_TOKEN` (nunca se versiona).

## Fase 3 — Generación fundamentada (MVP)

Cierra el primer ciclo extremo a extremo: pregunta → retrieval denso → evidencias acotadas →
prompt restrictivo → **LLM local (Ollama)** → JSON estructurado validado → respuesta ciudadana con
citas oficiales **o abstención**. Las URL y etiquetas finales provienen del corpus (datos
autoritativos), nunca del texto generado; el aviso jurídico se añade de forma estática.

**Dependencia externa (solo en ejecución real):** un Ollama local en `127.0.0.1:11434` con el
modelo configurado (generador del TFG: `qwen2.5:7b-instruct`; juez de evaluación `gemma3:12b`, de
familia distinta). No hace falta para los tests (offline).

Configuración en `.env` (ver `.env.example`): `OLLAMA_*` (URL loopback por defecto, modelo,
timeout, `keep_alive`, `num_ctx`, etc.) y `GENERATION_*` (bundle denso, perfil de consulta,
`top_k`, nº de evidencias, estrategia y presupuestos de contexto). El **bundle no es un default
global**: se indica con `--bundle` o `GENERATION_DENSE_BUNDLE`.

```bash
uv run python scripts/answer_question.py \
  --bundle data/indexes/dense/<bundle_id> \
  --query "¿Qué plazo tengo para interponer un recurso de alzada?"

# salida JSON completa y descarga del modelo de RAM al terminar
uv run python scripts/answer_question.py --bundle data/indexes/dense/<bundle_id> \
  --query "..." --json --unload-model
```

Sin evidencia suficiente, el sistema **se abstiene** en lugar de inventar. El exit code es ≠0 solo
ante fallo técnico (bundle inválido, Ollama caído, contrato del LLM incumplido); una abstención
válida no es un error.

**Tests:** la suite normal (`uv run pytest`) corre **offline**: sin red, sin Ollama, sin pesos
reales y sin bundle real (usa fakes y un bundle temporal sintético). La prueba **real** contra
Ollama está desactivada por defecto y se ejecuta en el servidor:

```bash
RUN_OLLAMA_INTEGRATION=1 uv run --locked pytest tests/test_integration_ollama.py -q -s
```

---

Repositorio **privado** (TFG en curso). © 2026 Jorge Bailez — Todos los derechos reservados.
