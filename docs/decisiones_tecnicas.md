# Decisiones técnicas

Registro de decisiones tomadas y alternativas descartadas. Documento vivo.

## Estado: Fase 2 (índice denso) implementada — dense-only

Cerrado el flujo local end-to-end sobre 10 normas (descarga → manifest → parser → chunking →
auditoría con *gate* previo a embeddings) y construida la **Fase 2 dense-only** (preparación de
inputs, encoder, bundle inmutable, índice exacto, consulta, evaluación). Las cargas pesadas
(descarga de modelos + codificación) se ejecutan en el servidor CPU. Pendiente: retrieval híbrido,
generación con LLM, API (fases posteriores).

## Arquitectura de contratos de datos (separación de responsabilidades)

- **Cuatro artefactos por norma, propiedad única del texto pesado.** El parser produce una
  **representación intermedia neutral** (no persistida) y de ella derivan:
  `boe_legal_document_v2` (**descriptor** legible, sin texto vigente), `boe_legal_history_v2`
  (versiones + notas de modificación + `temporal_resolution`, **un registro por cada `block_id`**),
  `boe_legal_parents_v2` (**propietario único** del texto vigente y los párrafos) y
  `boe_legal_chunks_v2` (vector-ready: `text`, `retrieval_text`, `citation`, `filters`; **sin**
  texto del padre ni metadatos documentales). La **fuente procesada autoritativa es compuesta**
  (`documents + histories + parents`); `chunks`/`reports` son proyecciones.
- **Cobertura de parents = todo bloque resuelto con texto vigente no vacío** (independiente de
  indexabilidad): incluye encabezados con texto, firmas y notas iniciales. Indexable ⇒ genera
  chunks; resuelto no indexable ⇒ parent sin chunks; cuarentena ⇒ sin parent (history conserva
  diagnóstico). En el corpus: 2.256 bloques → 2.256 parents, de los cuales 452 sin chunks.
- **Propiedad / proyecciones.** Texto vigente + párrafos → solo `parents`; versiones + notas de
  modificación + resolución temporal → solo `histories`; materias completas → solo
  `documents.analysis.subjects` (en chunks solo `subject_codes`); `indexable`/`excluded_reason`
  → solo `documents.blocks[]`; `retrieval_text` → solo lo construye y persiste el chunker;
  `documents.source.manifest_ref` es ruta **relativa**. Diagnósticos pesados (findings, métricas,
  readiness) → solo `reports/`; en `documents`/`chunks` solo `generation_meta` mínimo
  (`generated_at`, `generator`).
- **Parser desacoplado de la persistencia.** `_build_normalized_intermediate` ensambla el modelo
  de trabajo; `build_processed_bundle → ProcessedNormBundle` deriva los tres contratos (y los
  valida con Pydantic); `save_processed_bundle` los escribe. El chunker consume `document + parents`.
- **Validación en dos capas.** Pydantic v2 (`src/contracts/`, `extra="forbid"`,
  `schema_version: Literal[...]`) valida **cada artefacto** (en runtime al persistir y en
  `validate_mvp_corpus.py`); la auditoría (`corpus_audit`) valida lo **relacional** (joins,
  cobertura, ownership, `current_version` ↔ history). `schemas/*.json` se exportan de forma
  **determinista** desde los modelos (test anti-drift).
- **Eficiencia (medida):** chunks 13,19 MB (sin `parent_text` ni `subjects` replicados; el texto
  vigente vive una sola vez en `parents`, 11,40 MB).

## Decisiones de ingesta raw

- **Guardar raw + manifest antes de parsear.** El cliente (`src/boe/client.py`) descarga
  los endpoints de la norma y los guarda como bytes (`<endpoint>.xml`) sin interpretarlos,
  más un manifest JSON (`data/manifests/<id>.json`) con `sha256` y `size_bytes` por fichero.
  Motivo: trazabilidad y reproducibilidad (poder re-descargar y comparar hashes) y desacople
  entre ingesta y parsing. El parser trabajará sobre el raw guardado, no sobre la red.
- **Cliente HTTP con `httpx`** (no `requests`): API moderna, cliente inyectable para tests
  vía `httpx.MockTransport` (sin red en CI). No se hace scraping HTML ni parsing de XML aquí.
- **Errores de dominio.** `BoeApiError` para fallos HTTP (status no exitoso) y de red/timeout,
  incluyendo status y endpoint en el mensaje. `ValueError` para identificadores mal formados
  (validación de entrada `BOE-A-YYYY-NNNNN`), distinguiendo error de cliente de error de API.
- **Manifest versionado, raw no.** Los XML descargados no se versionan (`.gitignore`); el
  manifest sí, como evidencia de qué se descargó y con qué hash.

## Decisiones del parser v0

- **El parser v0 usa los ficheros separados** (`metadatos.xml`, `analisis.xml`,
  `indice.xml`, `texto.xml`) + el manifest, **no `full.xml`**. Motivo: `indice.xml` (URL y
  fecha por bloque) no está en `full.xml`, y los ficheros separados se alinean 1:1 con el
  manifest y sus `sha256`. `full.xml` queda como respaldo; `metadata_eli.xml` no se parsea en
  v0 (redundante con `metadatos`). Ver `docs/modelo_documental.md`.
- **`latest_version` con texto; `versions[]` solo metadatos.** Cada bloque puede tener varias
  `<version>`; **la vigente se selecciona por fecha, no por orden XML** (ver "Integridad
  temporal"). Solo `latest_version` lleva `text`/`paragraphs`; el historial de versiones se
  guarda como metadatos (fechas + `source_norm_id`) sin duplicar texto. El MVP indexa solo la
  versión vigente.
- **Notas de modificación fuera del texto normativo.** Los `<p class="nota_pie/nota_pie_2">`
  (dentro de `<blockquote>`) se extraen como notas de modificación (con `target_norm_id` por
  regex `BOE-A-...`), se persisten en `histories` (propietario) y **no** entran en `retrieval_text`
  ni en los chunks.
- **`lxml`** para el parseo (ya en dependencias). Sin red, sin nuevas dependencias.

## Decisiones del chunking

- **Parent-child chunking por párrafos.** El bloque jurídico (`boe_block`) es el documento
  padre: cada chunk lleva `parent_id` y el texto del padre se resuelve por **join** a `parents`
  (no se replica en el chunk). El troceado agrupa los **párrafos** del bloque en orden hasta
  `max_chars=1800`, sin cortes arbitrarios de caracteres y sin reescribir el texto legal. Esquema
  de salida: `boe_legal_chunks_v2` en `data/processed/chunks/<norm_id>.json` (no versionado);
  estrategia `legal_parent_child_paragraphs`.
- **Overlap de 1 párrafo** entre chunks consecutivos **solo cuando un bloque se divide** en
  varios chunks (continuidad de contexto sin duplicar bloques que caben en uno).
- **Solo bloques indexables** (`retrieval.indexable == true`): artículos/disposiciones y
  preámbulo (marcado con `is_preamble=true`). Se excluyen `encabezado` y `firma`.
- **Párrafo único > `max_chars`**: se emite como un chunk que supera el límite y se registra
  en `quality_checks.oversized_chunks` (no se parte un párrafo legal por la mitad).
- **`modification_notes` fuera de los chunks**: se trocean los `paragraphs` normativos, no el
  texto con notas; `retrieval_text` antepone contexto (norma + jerarquía + título del bloque)
  sin alterar el texto legal.
- **Corrección de `citation_label`**: `_lower_first` en el parser solo minúscula la inicial de
  palabras *title-case* (`Artículo`→`artículo`), dejando intactos `TÍTULO`/`CAPÍTULO`.

## Decisiones del corpus MVP

- **Corpus semilla de 10 normas** (`data/corpus/seed_corpus.json`, versionado) en torno al
  derecho administrativo/sector público, para poder probar el recuperador con variedad real
  (no una sola norma). Es la fuente única de verdad para el script y el notebook.
- **Verificación con criterios y reporte, sin sustitución silenciosa.** `scripts/build_corpus.py`
  descarga, verifica y procesa cada norma. Criterio de inclusión: **vigente**
  (`estatus_derogacion == "N"` y `vigencia_agotada == "N"`) **y** `estado_consolidacion ==
  "Finalizado"` **y** endpoints obligatorios (`metadatos`, `texto`, `indice`) disponibles. Las
  que no cumplen se excluyen y se reportan en `data/corpus/verification_report.json`; la
  sustitución por una alternativa requiere aprobación explícita.
- **Endpoints opcionales tolerados.** `BoeClient.download_norm_raw(optional_endpoints=...)`
  omite `analisis`/`metadata_eli`/`full` si fallan (no abortan la descarga); el parser tolera
  `analisis.xml` ausente (análisis vacío). Los obligatorios siguen siendo estrictos.
- **Verificación sin red.** Se evalúa desde el raw ya descargado (reutilizando
  `parse_metadata`), evitando una segunda descarga del `texto` (que puede ser grande).
- **Notebook narrativo** (`notebooks/01_exploracion_api_boe.ipynb`) como documentación viva
  del recorrido; la lógica reutilizable vive en `src/` para mantener el notebook limpio.

## Decisiones de la corrección previa a embeddings

- **Indexabilidad por contenido, no por tipo.** `retrieval.indexable = has_retrievable_body and
  block_type ∉ {firma, nota_inicial}`. `has_retrievable_body` = el bloque tiene ≥1 párrafo no
  estructural (función reusable `heading_has_retrievable_body`). Motivo: los anexos y otro
  contenido normativo llegan como `tipo=encabezado` con cuerpo; la regla por tipo los dejaba
  fuera de retrieval. Ahora se indexan 10 encabezados sustantivos (anexos, `dd`, `a3-2`, `ciii-5`).
- **`nota_inicial` no indexable** pero conservado en el documento (editorial: "Incluye la
  corrección de errores…"), para trazabilidad.
- **Roles semánticos aditivos** por bloque: `semantic_role` (precept/preamble/signature/
  initial_note/annex/content_heading/structural_heading), `has_retrievable_body`, `is_annex`,
  `contains_table`, `table_text_available`, `contains_image`. `block_type` se mantiene como
  valor **raw** del BOE. Estos flags viven en el descriptor (`boe_legal_document_v2`).
- **Jerarquía a 6 niveles** `{book, title, chapter, section, subsection, annex}` con reinicio de
  inferiores y `annex` (mutuamente excluyentes). Solo clases inequívocas (`*_num`/`seccion`/
  `subseccion`) actualizan el estado; las singulares (`titulo`/`libro`/`anexo`) no se propagan
  (evitan envenenar el cuerpo). Un `anexo` singular que sí es anexo recibe `hierarchy.annex`
  **local**. `_full_title` se extrae del prefijo estructural inicial (no del cuerpo).
- **`retrieval_text` sin `..`**: el prefijo de contexto se une con separador inteligente (`" "`
  tras `. : ; ? !`, si no `". "`). **Dedup condicional** de la cabecera: si el texto del chunk
  ya empieza por `full_title`/`block_title`, no se repite en el prefijo. No se altera `chunk.text`.
- **Imágenes y tablas**: `contains_image` es metadata (no vuelve indexable por sí solo; sin OCR);
  las tablas se detectan por `<table>` o clases `cabeza_tabla`/`cuerpo_tabla_*` y su texto
  linealizado (`table_text_available`) cuenta como cuerpo recuperable.
- **Catálogo de corpus único**: `data/corpus/seed_corpus.json` (sin duplicar en `config/`). Los
  scripts locales `process_mvp_corpus.py` y `validate_mvp_corpus.py` (sin red) lo reutilizan;
  `build_corpus.py` (con red) queda para la adquisición.
- **H3 (oversized) diferido a indexación**: no se parten párrafos; la decisión se toma midiendo
  **tokens** sobre `retrieval_text` con el tokenizador y `max_seq_length` del embeddings elegido.

## Decisiones de integridad temporal (vigencia)

- **La vigencia la decide el índice, no el orden XML.** El parser ya **no** usa
  `version_elements[-1]`: la versión vigente es la única cuya `fecha_publicacion` coincide de
  forma **exacta** con la `fecha_actualizacion` del bloque en `indice.xml` **y** que además es
  la `fecha_publicacion` **máxima** normalizable. Motivo: el XML del BOE no garantiza orden
  cronológico de `<version>` (en el corpus, 7 bloques no cronológicos; 4 indexaban texto
  histórico con el criterio antiguo, p. ej. LBRL `a45` servía la redacción de 1990 en lugar de
  la vigente de 2013 «(Sin contenido)»). `resolve_current_version` es la función canónica,
  compartida por parser y auditoría (esta última recalcula de forma **independiente** desde
  `versions[]` + `index_last_update_date`, sin confiar en lo persistido).
- **Cuarentena temporal sin fallback silencioso.** Si la resolución no es única
  (`unresolved`/`ambiguous`/`missing_index_date`/`invalid_date`/`index_not_max`), el bloque
  conserva `versions[]` para diagnóstico pero queda en cuarentena: `latest_version = null`,
  `retrieval.indexable = false`, `retrieval.excluded_reason = "temporal_quarantine:<status>"`,
  **sin chunks**. El batch continúa (modo diagnóstico) pero el *gate* bloquea. **Nunca** se cae
  a `versions[-1]` ni a `max(fecha_publicacion)` para retrieval (`max` es solo diagnóstico). En
  el corpus actual, 0 bloques en cuarentena (los 2.256 resuelven limpio).
- **Bloques vigentes «(Sin contenido)» sí se indexan.** Un artículo vaciado (p. ej. por una
  reforma) se conserva e indexa como chunk **informativo** (mantiene cita + URL oficial), con
  metadata **neutral** `content_status="without_content"`/`is_without_content=true`. **No** se
  infiere derogación/causa/norma salvo evidencia trazable en `analysis`/`modification_notes`.
- **Entrada en vigor futura**: si la versión vigente tiene `validity_date > processing_date` se
  reporta en `temporal_integrity.future_effective_selected_versions` (política explícita: es la
  redacción consolidada a la que apunta el índice), pero **no** bloquea el MVP (hoy 0 casos).
- **Gate ampliado.** `pre_embedding_readiness` integra ahora `temporal_integrity` y
  `raw_integrity` (sha256/size del raw vs manifests) además de la integridad estructural.
  Modos: batch (informe completo) vs `--strict` (`exit≠0` ante bloqueante) en
  `validate_mvp_corpus.py` y `audit_corpus.py`; `validate_raw_integrity.py` falla si algún hash
  o tamaño no coincide.

## Decisiones de arranque

- **Gestión de entorno y dependencias con `uv`** (no `pip`/`venv` directos).
  Motivo: reproducibilidad y velocidad; alineado con `CLAUDE.md`.
- **Layout `src/` como namespace package** (sin `src/__init__.py`), importable como
  `src.<paquete>.<modulo>`. Proyecto no empaquetado en esta fase.
- **Configuración con `pydantic-settings`** y valores por defecto seguros, de modo que
  `Settings()` se importe sin `.env` ni secretos.
- **Dependencias pesadas pospuestas**: LlamaIndex, embeddings, vector stores, FastAPI,
  Ollama y RAGAS se documentan como futuras y se añadirán cuando llegue su fase.

## Decisiones de Fase 2 (índice denso, dense-only)

- **Dense-only como baseline.** Se implementa solo recuperación densa exacta; BM25, híbrido,
  reranking y generación quedan para fases posteriores. El valor es la trazabilidad y la evaluación,
  no la escala. Sin abstracciones para backends inexistentes.
- **Índice exacto, no ANN ni vector DB.** `ExactDenseIndex` hace producto escalar sobre vectores
  L2-normalizados con la matriz `embeddings.npy` por mmap de solo lectura (`allow_pickle=False`).
  Sin Chroma/Qdrant en esta fase (se eliminaron sus defaults de configuración).
- **El modelo no es un default global.** Se elige con `--model` (alias corto). El registro
  (`model_registry.py`) separa `document_embedding_contract` (identidad de los embeddings) de
  `query_profile` (configurable). Las revisiones (commit hash) se dejan **sin fijar**: el gate
  bloquea hasta fijarlas o usar `--allow-unpinned-revision`; nunca `revision="main"` silencioso.
- **Truncamiento silencioso prohibido.** Antes de codificar se cuenta con el tokenizador real
  (special tokens incluidos) y, si el input excede el límite, se **repara** en ventanas token-aware
  (overlap 100) conservando trazabilidad. Si quedara overflow sin reparar, error bloqueante. Esto
  cierra H3 (medición por tokens en indexación).
- **Vistas J1/J2/C1.** J1 (`retrieval_text`) es el baseline; J2 (`text`) y C1 (ventanas fijas dentro
  del parent) son ablaciones. Los chunks oficiales de Fase 1 no se modifican; el texto nuevo
  derivado se persiste en las rows.
- **Bundle inmutable + gates.** Publicación segura (staging → validar → checksums → manifest →
  rename atómico). Gate A (pre-encoding) y Gate B (pre-publicación) con severidades ERROR/WARNING/
  INFO. Fingerprints distintos (corpus / inputs / contrato) sobre JSON canónico SHA-256.
- **HF_TOKEN solo del entorno.** Nunca se incrusta, imprime ni persiste; opcional (solo si el modelo
  requiere autenticación).
- **Evaluación.** Métrica primaria ParentnDCG@10 + controles; bootstrap pareado con semilla fija.
  Dataset versionable con Gate C que bloquea el benchmark formal hasta la anotación revisada.

## Alternativas descartadas (de momento)

- Empaquetar el proyecto (`[build-system]` + instalación editable): innecesario para el
  MVP; se reconsiderará si se publica como paquete.

## Pendientes de decisión

Ver `known_issues.md`.
