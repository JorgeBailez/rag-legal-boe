# RAG Legal BOE (TFG)

Sistema de recuperación y generación (RAG) para consultar de forma informativa la legislación
consolidada del BOE, pensado para ciudadanos no expertos. No es asesoramiento jurídico vinculante: los
textos consolidados del BOE tienen carácter informativo y no valor jurídico oficial.

El sistema funciona de principio a fin sobre 92 normas. Descarga el XML original del BOE y lo conserva
inmutable, lo parsea en una representación por capas (descriptor, historial y texto vigente), lo trocea
en fragmentos recuperables y lo audita. Sobre esa base genera los embeddings densos como un paquete
inmutable, construye un índice exacto, recupera y ensambla el contexto, y produce una respuesta
fundamentada con un modelo de lenguaje local. La respuesta se devuelve como JSON validado con citas
oficiales, o el sistema se abstiene si no hay evidencia suficiente.

Resultados principales, con la evidencia y los números en
[`docs/decisiones_de_diseno.md`](docs/decisiones_de_diseno.md): la recuperación densa
(`e5-large-instruct`) es la mejor opción en este corpus y combinarla con BM25 no la mejora; la
generación es deliberadamente conservadora, es decir, prefiere abstenerse de más antes que responder a
una pregunta fuera del corpus; y el juez automático se validó contra anotación humana y resultó
insuficiente, así que la fidelidad y la corrección se reportan con anotación humana. La interfaz web y
la API quedan como trabajo futuro.

Un detalle de diseño que conviene conocer: la versión vigente de cada bloque se decide por la fecha de
actualización del índice de la norma (la coincidencia exacta y única con una versión, que además es la
de fecha máxima), nunca por el orden en que aparecen las versiones en el XML. Los bloques que no se
pueden resolver van a cuarentena (no se indexan) y detienen el avance hacia los embeddings.

Los embeddings, los paquetes de índice y los pesos de los modelos no se versionan (ver `.gitignore`):
se generan en local.

## Requisitos

- Python 3.11 o superior.
- [`uv`](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias.

Para instalar `uv`:

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

No hacen falta secretos para arrancar: la configuración por defecto funciona sin tocar nada. La suite
de tests corre sin red, sin Ollama y sin pesos reales.

## Estructura del repositorio

```
src/        Código fuente (boe, preprocessing, indexing, retrieval,
            generation, evaluation, app, core, config)
data/       Datos (raw, processed, evaluation, manifests). El contenido pesado no se versiona;
            sí se versionan el catálogo de normas, los manifests y los informes de verificación
notebooks/  Cuadernos que narran el recorrido del pipeline
prompts/    Plantillas de prompt
tests/      Suite de pytest
docs/       Documentación de diseño y análisis (el subconjunto público; ver docs/decisiones_de_diseno.md)
thesis/     Memoria del TFG
```

El pipeline se ejecuta norma a norma o sobre el corpus entero. Las secciones siguientes describen cada
paso y su comando.

## Descargar el original del BOE

Descarga la respuesta del BOE para una norma consolidada, sin parsear, y genera un manifest con sus
hashes y tamaños para poder verificarla después:

```bash
uv run python scripts/download_boe_raw.py BOE-A-2015-10565
```

Genera:

```
data/raw/boe/BOE-A-2015-10565/     # full.xml, metadatos.xml, analisis.xml,
                                   # metadata_eli.xml, texto.xml, indice.xml
data/manifests/BOE-A-2015-10565.json
```

El XML descargado no se versiona; el manifest sí, como evidencia de reproducibilidad. La dirección de
la API se configura con `BOE_API_BASE` (ver `.env.example`). Este paso llama a la API externa del BOE.

## Parsear el original (XML a artefactos)

Convierte el original local de una norma ya descargada en una representación intermedia y deriva los
tres artefactos persistidos. No usa internet:

```bash
uv run python scripts/parse_boe_raw.py BOE-A-2015-10565
```

Lee `data/raw/boe/<norm_id>/{metadatos,analisis,indice,texto}.xml` y el manifest, y escribe el
descriptor de la norma, su historial y sus bloques padre (el propietario único del texto vigente) en
`data/processed/`.

## Contratos de datos

La representación procesada está repartida en piezas con propiedad única del texto: el descriptor
(legible), el historial (versiones, notas de modificación y resolución temporal) y los bloques padre
(el texto vigente y sus párrafos, una sola vez). Los fragmentos son una proyección mínima para la
búsqueda, y los informes una proyección de auditoría. Los modelos viven en `src/contracts/` como fuente
única de verdad y generan los JSON Schema de `schemas/`, que se validan al persistir. Para ver una norma
ya procesada de forma legible:

```bash
uv run python scripts/inspect_processed_norm.py <id> [--block <bid>]
```

## Trocear (chunking) la norma

Convierte el descriptor y los bloques padre en fragmentos recuperables, troceando por párrafos y sin
red. El bloque jurídico actúa como padre del fragmento:

```bash
uv run python scripts/chunk_boe_document.py BOE-A-2015-10565
```

Solo se trocean los bloques indexables; los que no lo son (encabezados con texto, firmas, notas
iniciales) conservan su bloque padre pero no generan fragmentos. Cada fragmento lleva su bloque padre,
la cita, unos filtros mínimos y el texto de recuperación con contexto jurídico, pero no el texto del
padre ni los metadatos del documento, que se resuelven por unión.

## Corpus (92 normas)

El corpus son 92 normas, listadas en `data/corpus/seed_corpus_ampliado.json` (el catálogo de 10 normas
fue el prototipo inicial). Este comando descarga, verifica (que la norma esté vigente, en estado
"Finalizado" y con todos los endpoints) y procesa (parser y troceado) las normas que cumplen los
criterios:

```bash
uv run python scripts/build_corpus.py
```

Escribe un informe de verificación versionado e imprime una tabla. Las normas que no cumplen los
criterios se excluyen y se reportan; no se sustituyen de forma automática.

## Auditoría de calidad del corpus

Audita, sin modificar nada, que el parser y el troceado producen lo esperado: integridad, trazabilidad
del XML al fragmento, contexto de recuperación, solapes, fragmentos demasiado grandes, jerarquía y
eficiencia.

```bash
uv run python scripts/audit_corpus.py
```

Para regenerar el corpus en local desde el original ya descargado (sin red) y validarlo:

```bash
uv run python scripts/process_mvp_corpus.py        # regenera descriptor, historial, padres y fragmentos
uv run python scripts/validate_raw_integrity.py    # comprueba hashes y tamaños frente a los manifests
uv run python scripts/validate_mvp_corpus.py --strict  # valida los contratos y la integridad relacional
uv run python scripts/audit_corpus.py --strict     # termina con error si el corpus no está listo
```

El corpus se considera listo para los embeddings solo si no hay errores estructurales, ni divergencias
de vigencia, la integridad del original es correcta, no se filtran notas editoriales y no queda
contenido sustantivo fuera de la búsqueda.

## Cuaderno de exploración

`notebooks/01_exploracion_api_boe.ipynb` documenta el recorrido y las decisiones, de la API al corpus.
Requiere las dependencias de desarrollo (`uv sync`) y se ejecuta a mano:

```bash
uv run jupyter notebook notebooks/01_exploracion_api_boe.ipynb
```

## Índice denso y recuperación

Embeddings densos reproducibles, índice exacto, consulta y evaluación, a partir de los fragmentos de
`data/processed/`. El índice es denso (numpy con memoria mapeada). La comparación con BM25 y con la
fusión híbrida se ejecuta aparte, con `scripts/benchmark_retrieval_strategies.py`, y concluyó que la
recuperación densa gana en este corpus (el detalle está en `docs/decisiones_de_diseno.md`). El modelo de
embeddings se elige de forma explícita con `--model`, no hay uno por defecto:

```bash
uv run python scripts/generate_dense_index.py --list-models        # modelos disponibles
uv run python scripts/generate_dense_index.py --model bge-m3 --preflight-only
uv run python scripts/generate_dense_index.py --model bge-m3       # genera el paquete de índice
uv run python scripts/validate_dense_index.py --bundle data/indexes/dense/<bundle_id>
uv run python scripts/query_dense_index.py --bundle data/indexes/dense/<bundle_id> \
  --query "¿Cuánto tiempo tiene la Administración para responder a mi solicitud?"
```

Las cargas pesadas (descargar los modelos y codificar) se hacen en un servidor; el código se valida sin
red con datos de prueba. Los paquetes de índice publicados son inmutables y exigen fijar la revisión
exacta del modelo y del tokenizador. El diseño y las decisiones del índice están en
[`docs/fase2_dense_baseline.md`](docs/fase2_dense_baseline.md). La autenticación con Hugging Face es
opcional, mediante la variable de entorno `HF_TOKEN`, que nunca se versiona.

## Generación de respuestas

Cierra el ciclo de extremo a extremo: pregunta, recuperación densa, evidencias acotadas, prompt
restrictivo, modelo de lenguaje local (Ollama), JSON validado y, por último, una respuesta para el
ciudadano con citas oficiales o una abstención. Las direcciones y etiquetas finales salen del corpus,
nunca del texto generado, y el aviso jurídico se añade de forma fija.

Esto requiere, solo en ejecución real, un Ollama local en `127.0.0.1:11434` con el modelo configurado
(el generador del trabajo es `qwen2.5:7b-instruct` y el juez de evaluación, de una familia distinta,
`gemma3:12b`). Los tests no lo necesitan.

La configuración va en `.env` (ver `.env.example`): las variables `OLLAMA_*` (dirección, modelo,
tiempos) y `GENERATION_*` (paquete de índice, perfil de consulta, número de resultados y evidencias,
estrategia y presupuesto de contexto). El paquete de índice no tiene valor por defecto: se indica con
`--bundle` o con `GENERATION_DENSE_BUNDLE`.

```bash
uv run python scripts/answer_question.py \
  --bundle data/indexes/dense/<bundle_id> \
  --query "¿Qué plazo tengo para interponer un recurso de alzada?"

# salida JSON completa y descarga del modelo de la memoria al terminar
uv run python scripts/answer_question.py --bundle data/indexes/dense/<bundle_id> \
  --query "..." --json --unload-model
```

Sin evidencia suficiente, el sistema se abstiene en lugar de inventar. Una abstención válida no es un
error: el código termina con error solo ante un fallo técnico (un paquete de índice inválido, Ollama
caído o una respuesta del modelo que incumple el contrato).

La prueba real contra Ollama está desactivada por defecto y se ejecuta en el servidor:

```bash
RUN_OLLAMA_INTEGRATION=1 uv run --locked pytest tests/test_integration_ollama.py -q -s
```

---

Trabajo de Fin de Grado. © 2026 Jorge Bailez Martínez.
