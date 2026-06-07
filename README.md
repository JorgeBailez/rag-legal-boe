# RAG Legal BOE (TFG)

Sistema RAG para consulta **informativa** de legislación consolidada del BOE, orientado a
ciudadanos no expertos. **No** es asesoramiento jurídico vinculante: los textos
consolidados del BOE tienen carácter informativo y no valor jurídico oficial.

> **Estado: Fase 2 densa cerrada a checkpoint · Fase 3 (generación MVP) implementada.**
> Flujo extremo a extremo sobre 10 normas: raw XML inmutable → manifest → parser →
> `documents + histories + parents` → chunks vector-ready → auditoría con *gate* →
> **embeddings densos reproducibles → bundle inmutable → índice exacto (numpy + mmap) →
> retrieval dense-only evaluado** → **generación fundamentada con LLM local (Ollama) → JSON
> estructurado validado con citas oficiales o abstención**. Falta la API (FastAPI), prevista
> para una fase posterior.
>
> El baseline denso seleccionado (checkpoint) es `e5-large-instruct · vista J1 ·
> I2_CITIZEN_LEGISLATION`. Los bundles densos y los pesos de los modelos **no se versionan**
> (ver `.gitignore`): se generan en un servidor CPU.
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

## Corpus MVP (10 normas)

El corpus semilla está en `data/corpus/seed_corpus.json`. Este comando descarga, **verifica**
(vigente + `estado_consolidacion = Finalizado` + endpoints obligatorios) y procesa
(parser + chunking) todas las normas que cumplen criterios. Llama a la API del BOE:

```bash
uv run python scripts/build_corpus.py
```

Escribe `data/corpus/verification_report.json` (versionado, como evidencia) e imprime una
tabla. Las normas que **no** cumplen criterios se excluyen del procesado y se reportan —
**no se sustituyen automáticamente**.

## Auditoría de calidad del corpus

Audita (solo lectura) que parser y chunker generan lo esperado sobre las 10 normas
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
`data/processed/chunks/<norm_id>.json`. **Dense-only**: BM25, híbrido y reranking quedan para fases
posteriores (no son parte de esta fase). El modelo de embeddings **no** es un default global: se
elige con `--model`.

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
modelo configurado (por defecto `qwen3:4b-instruct`). No hace falta para los tests (offline).

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
