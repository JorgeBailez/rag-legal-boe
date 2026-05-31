# RAG Legal BOE (TFG)

Sistema RAG para consulta **informativa** de legislación consolidada del BOE, orientado a
ciudadanos no expertos. **No** es asesoramiento jurídico vinculante: los textos
consolidados del BOE tienen carácter informativo y no valor jurídico oficial.

> **Estado: corpus MVP listo para indexar.** Cerrado el flujo local sobre 10 normas:
> descarga raw → manifest → parser (`boe_legal_document_v1`) → chunking jurídico
> (`boe_legal_chunks_v1`) → auditoría con *gate* previo a embeddings (integridad estructural,
> **temporal** y de *raw*). Todavía no hay embeddings, retrieval, generación ni API.
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

## Parseo BOE (raw → JSON)

Convierte el raw local de una norma ya descargada en el modelo documental
`boe_legal_document_v1`. No llama a internet:

```bash
uv run python scripts/parse_boe_raw.py BOE-A-2015-10565
```

Lee `data/raw/boe/<norm_id>/{metadatos,analisis,indice,texto}.xml` +
`data/manifests/<norm_id>.json` y escribe `data/processed/documents/<norm_id>.json`
(tampoco se versiona). No usa `full.xml` como fuente ni parsea `metadata_eli.xml`.

## Chunking jurídico

Convierte el documento procesado en chunks recuperables `boe_legal_chunks_v1`
(parent-child: el bloque jurídico es el padre). Chunking por párrafos, sin red:

```bash
uv run python scripts/chunk_boe_document.py BOE-A-2015-10565
```

Lee `data/processed/documents/<norm_id>.json` y escribe
`data/processed/chunks/<norm_id>.json` (no se versiona). Solo genera chunks de bloques
indexables (artículos/disposiciones/preámbulo); excluye encabezados y firma. Cada chunk
conserva `parent_id`, `parent_text`, `citation_label` y `source_url`, y un `retrieval_text`
con contexto jurídico. Aún no hace embeddings ni indexación.

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
uv run python scripts/process_mvp_corpus.py        # regenera documents/ y chunks/ de las 10
uv run python scripts/validate_raw_integrity.py    # sha256/size del raw vs manifests
uv run python scripts/validate_mvp_corpus.py --strict  # 0 ERROR (y 0 WARN en estricto)
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

## Siguiente tarea

Implementar la **indexación** (`src/indexing/`): embeddings + búsqueda semántica (y BM25),
partiendo de `data/processed/chunks/<norm_id>.json`.

---

Repositorio **privado** (TFG en curso). © 2026 Jorge Bailez — Todos los derechos reservados.
