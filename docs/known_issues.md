# Known issues / pendientes

Lista de puntos abiertos a confirmar antes de avanzar. Documento vivo.

## Deuda técnica

- **mypy no está configurado a nivel de repo.** `uv run mypy src` falla con
  "Source file found twice under different module names" (layout `src/` namespace sin
  `__init__.py` raíz). Con `--explicit-package-bases` los módulos type-checkean. Pendiente:
  decidir config mypy (`explicit_package_bases`/`mypy_path`) o `packages` en pyproject.
- **Anotación menor en `parser._update_hierarchy`**: el `state` se pasa como `dict | None` en una
  llamada aunque el guard lo protege en runtime. No afecta a la ejecución.
- **`reports/` no se versiona** (artefactos generados): `mvp_chunking_audit.*` y
  `tokenizer_profile.*` se regeneran con los scripts.

## Pendientes funcionales

- [x] ~~Confirmar los **endpoints exactos** de legislación consolidada del BOE antes de
      implementar `src/boe/client.py`~~ **Resuelto**: 6 endpoints confirmados e implementados
      (`full`, `metadatos`, `analisis`, `metadata_eli`, `texto`, `indice`); ver
      `fuentes_y_licencias.md`.
- [x] ~~Confirmar si el MVP usará **Chroma**/**Qdrant**~~ **Resuelto (Fase 2)**: el baseline es
      dense-only con **índice exacto** (numpy + mmap, sin servicio externo). Se eliminaron de la
      configuración los defaults `vector_store_provider`/`chroma_persist_dir`/`qdrant_*`. Chroma/
      Qdrant quedan como posible ruta de migración futura, **no** parte de esta fase.
- [ ] Confirmar el **modelo local de Ollama** cuando llegue la fase de generación
      (default actual: `mistral`).
- [ ] Confirmar la **estrategia de evaluación** antes de añadir RAGAS.
- [ ] Definir el **umbral de abstención** sobre el dataset real.
- [ ] Evaluar **embeddings en español** (e5-base/large/large-instruct, bge-m3, qwen3-0.6b,
      gte-multilingual-base): shortlist en `src/embeddings/model_registry.py`. **Ya no hay default
      global** de embeddings; el modelo se elige con `--model` (Fase 2). Pendiente: fijar los commit
      hashes (revisiones) y ejecutar el benchmark en el servidor.

## Riesgos / pendientes del parser v0

- **Jerarquía aproximada en disposiciones.** `hierarchy` se infiere por orden documental
  arrastrando el último TÍTULO/CAPÍTULO/SECCIÓN visto. Las disposiciones (adicionales,
  transitorias, finales) heredan el último encabezado de la parte articulada, que puede no
  ser su contexto real. Suficiente para v0; revisar si se necesita jerarquía exacta.
- **`target_norm_id` por regex.** En `modification_notes` se extrae el primer
  `BOE-[A-Z]-\d{4}-\d+` del texto/enlace de la nota. No resuelve sub-anclas (`#a3`, `#df`) ni
  varias normas por nota (se toma la primera). Suficiente para trazabilidad básica.
- **Indexabilidad por contenido** (ya no por tipo). `retrieval.indexable = has_retrievable_body
  and block_type ∉ {firma, nota_inicial}`. Los encabezados con cuerpo (anexos, disposiciones o
  artículos embebidos en `tipo=encabezado`) **sí** se indexan; los rótulos puros no.
- **`metadata_eli.xml` sin parsear en v0** (RDF/ELI redundante); se conserva como raw para
  una fase posterior (grafo jurídico / modelo ELI de expresiones).
- [x] ~~`citation_label` de encabezados aparecía como `tÍTULO`/`cAPÍTULO`~~ **Resuelto**.
- **Jerarquía: clases-rótulo singulares no propagadas.** `titulo`/`libro` singulares dan
  `full_title` pero no actualizan el estado jerárquico (4 normas; limitación menor aceptada).

## Riesgos / pendientes del chunking

- **Overlap mínimo (1 párrafo).** Suficiente para continuidad en bloques largos; revisar el
  valor cuando se evalúe la recuperación sobre el dataset real.
- [x] ~~**Redundancia del índice** (`parent_text`/`subjects` repetidos por chunk)~~ **Resuelto**:
  los chunks v2 son vector-ready mínimos (sin texto del padre ni materias completas); el texto
  vigente vive una sola vez en `parents` y las materias en `documents.analysis.subjects`
  (en chunk solo `subject_codes`). Se resuelven por join.

## Hallazgos de la auditoría del corpus MVP (ver `docs/auditoria_chunking_mvp.md`)

Auditoría de solo lectura sobre las 10 normas (`uv run python scripts/audit_corpus.py`).
Tras la corrección previa a embeddings y la **corrección de integridad temporal**:
**`pre_embedding_readiness.ready = true`**, 0 ERROR, 0 WARN, 0 fugas de notas, overlap correcto,
`temporal_integrity.ready = true` y `raw_integrity.ready = true`.

- [x] ~~**Vigencia por orden XML (`version_elements[-1]`)**~~ **Resuelto**: selección por
      `indice.xml/fecha_actualizacion` (exacta+única+máxima); 4 bloques que indexaban texto
      histórico corregidos (`a2`/`a45` de BOE-A-1985-5392, `a7`/`a45` de BOE-A-2003-20977); 7
      bloques no cronológicos detectados y manejados; los irresolubles van a **cuarentena**
      (no indexables, sin chunks) y bloquean el *gate*. Ver `decisiones_tecnicas.md`.
- **Vigencia futura** (`validity_date > processing_date`): se reporta en
  `temporal_integrity.future_effective_selected_versions` (hoy 0); política explícita: no
  bloquea el MVP (es la consolidada que marca el índice).
- **Integridad raw**: `validate_raw_integrity.py` y `raw_integrity` en la auditoría comparan
  sha256/size del raw contra los manifests (60/60 OK). `raw_integrity.ready=false` bloquea el
  *gate*.
- [x] ~~**H1 — `nota_inicial` indexable**~~ **Resuelto**: `nota_inicial` excluido (5→0 indexables).
- [x] ~~**`..` en `retrieval_text`**~~ **Resuelto**: separador inteligente en el prefijo
      (2.923→0 artificiales; las `..` restantes son elipsis del texto legal, no del prefijo).
- [x] ~~**H2 — jerarquía sin LIBRO/SUBSECCIÓN/ANEXO**~~ **Resuelto**: `hierarchy` a 6 niveles;
      `headings_without_full_title = 0`.
- [x] ~~**Contenido normativo atrapado en `encabezado` no-anexo** (`dd`, `a3-2`, `ciii-5`)~~
      **Resuelto**: la indexabilidad por contenido los recupera (10 encabezados sustantivos
      indexados).
- [x] ~~**`full_title` duplicado** en `retrieval_text`~~ **Resuelto** (dedup condicional):
      1.784→1 (caso residual con repetición legítima de cabecera).
- [x] ~~**H3 — chunks oversized (89/3.271, máx. 3.550 car.).**~~ **Resuelto (Fase 2)**: el límite
      real es de **tokens**, no de caracteres. La preparación de inputs mide los tokens del input
      formateado con el tokenizador real del modelo y, si superan el límite efectivo, **repara** el
      input dividiéndolo en ventanas token-aware (overlap 100), sin partir en silencio. El
      truncamiento silencioso es imposible (error bloqueante si quedara overflow). Ver
      `src/embeddings/input_preparation.py` y `docs/fase2_dense_baseline.md`.
- [x] ~~**Eficiencia del índice** (`parent_text`/`subjects` replicados por vector)~~ **Resuelto**
      con los contratos v2 (chunks vector-ready mínimos; texto en `parents`, materias en `documents`).
- [ ] **(Deferred) imágenes y tablas sin texto.** No hay imágenes en el corpus; si una norma
      futura trae `<img>` sin texto alternativo o un `<table>` sin `<p>`, se reportan como
      `image_only_block_without_text` / `table_without_textual_representation` (no bloqueantes).

## Campaña de embeddings / bake-off (Fase C–E)

Estado tras la campaña en `dslab01` (2026-06-18): **4 bundles válidos** sobre el corpus limpio
(`e5-base`, `e5-large`, `e5-large-instruct`, `bge-m3`; Gate B OK, 3263–3301 vectores). Dos modelos
quedan fuera del bake-off activo:

- **(APLAZADO) `gte-multilingual-base`.** Su código remoto propio (`Alibaba-NLP/new-impl`,
  `modeling.py`, escrito para `transformers 4.39`) es **incompatible con el `transformers` del entorno
  actual** (el que exige `qwen3`): el encode revienta con
  `IndexError: ... rope_cos[position_ids]` (RoPE con `position_ids` corrupto), gatillado por el wrapper
  reciente `transformers/utils/output_capturing.py`. El **contrato es correcto** (pesos+tokenizer
  pinneados en `9bbca17d`, código remoto revisado el 2026-06-16), pero **no se puede ejecutar en este
  stack**. No bloquea el cierre del MVP. **Reactivar** con un venv de `transformers` pinneado a
  ~4.39–4.4x cuando se quiera su número. Decisión 2026-06-18 (opción "aplazar").
- **(Coste prohibitivo) `qwen3-0.6b`.** Funciona, pero en esta CPU (EPYC 7451 Naples) va a
  ~0,1–0,2 docs/s → horas por bundle. Se genera para completar la calidad del bake-off, pero su coste
  lo descalifica de facto para un despliegue CPU-only (hallazgo, no fallo).

**Caveat de medición:** `dslab01` es **compartido**; los tiempos/throughput **varían con la carga de
otros usuarios** (e5-large y e5-large-instruct, misma arquitectura, dieron ~6× de diferencia según la
ventana). La **calidad** (vectores deterministas, Gate B) NO se ve afectada; el **coste** solo es
fiable medido en ventana ociosa (`load < 1`) o se reporta con caveat explícito. Contendientes
prácticos en CPU: **familia e5 + bge-m3**.

## Inconsistencias detectadas en el scaffold (no bloqueantes)

- [x] ~~`CHROMA_PERSIST_DIR=data/indexes/chroma`...~~ **Resuelto (Fase 2)**: se eliminaron de
  `settings.py`/`.env.example` los defaults de Chroma/Qdrant. El bundle denso vive en
  `data/indexes/dense/<bundle_id>/` (runtime, no versionado).
- `src/core/` y `src/config/` no figuran en el mapa de repositorio de `CLAUDE.md`
  (que describe la configuración solo vía `.env`). Se han añadido en el scaffold.
- `data/manifests/` y la carpeta `prompts/` no aparecen en `CLAUDE.md`/`ARCHITECTURE.md`;
  son adiciones del scaffold. La plantilla de prompt vivirá en `generation/prompt.py`
  apoyándose en los `.txt` de `prompts/`.
