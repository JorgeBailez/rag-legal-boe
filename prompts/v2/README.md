# Prompts v2 — Experimento P2 nº1 (reducir la sobre-abstención)

Variante de los prompts de generación para un **A/B controlado** contra el baseline
(`prompts/system_prompt.txt` + `prompts/rag_prompt.txt`). Mismo retrieval, mismo generador
(`qwen2.5:7b-instruct`), misma semilla/temperatura; **solo cambian los dos prompts**.

## Hipótesis
El baseline es *fail-safe pero sobre-conservador*: en dev abstuvo en **7/39** preguntas respondibles
(~18 %), siempre con el motivo *"no hay información específica/explícita"*. La causa es el gatillo
difuso *"información suficiente para responder con fiabilidad"*. El v2 lo sustituye por una **regla de
decisión binaria** sin perder el fail-closed.

## Qué cambia (vs baseline)
- **Regla de decisión:** responder si el dato **está, se extrae o se combina** de las evidencias;
  abstenerse **solo** si el dato no aparece en ninguna. Se elimina "fiabilidad/específico/explícito"
  como motivo de abstención.
- **Candado fail-closed (TEST POR AFIRMACIÓN):** toda afirmación debe poder señalar una frase de una
  evidencia citada; prohibido aportar datos que no figuren. → mantiene la fidelidad alta.
- **SILENCIO ≠ CONTRADICCIÓN** + **CUÁNDO SÍ ABSTENERSE** (q0020 norma presente/dato ausente; q0051
  "(Sin contenido)"): contrapeso anti-sobreinferencia para que `false_answer` siga en 0.
- Secciones específicas por modo de fallo: **marco de la pregunta** (q0002), **extracción mínima**
  (q0013/q0016), **comparaciones** (q0032), **premisa falsa** en forma afirmativa-mínima (q0019),
  **pregunta ambigua** (q0039).

## Cómo ejecutarlo
```bash
uv run --locked python scripts/run_generation_eval.py \
  --bundle data/indexes/dense/e5-large-instruct__j1__bc11142bdcc5 \
  --split development --judge-model gemma3:12b \
  --prompts-dir prompts/v2          # baseline = misma orden SIN este flag
```
El report registra `prompts_dir` + `prompt_fingerprint`, así que baseline y v2 tienen `run_id`
distintos y comparables. Compara los dos reports en el notebook 06.

## Criterio de aceptación
El v2 se acepta si: **over_abstention baja de forma neta** Y **`false_answer` sigue en 0** Y la
**fidelidad no baja** en las preguntas que pasaron a `answered=true` (confirmar con κ del juez
validado o inspección manual; ver `docs/evaluacion_gold_y_metricas.md`). Si sube cualquier
`false_answer`, se rechaza.

## Riesgos residuales
- **q0019 (premisa falsa):** el modelo podría emitir la negación ("no son tres meses") como
  afirmación no soportada → el juez de fidelidad podría bajar L3. Vigilar por inspección.
- **Coste:** el system v2 es más largo (más prompt_eval/consulta en CPU). Smoke con `--limit` antes.
- **Depende del retrieval:** si el parent esperado no entra en las evidencias, ningún prompt lo
  rescata (revisar `retrieved_parents`/`omitted_evidences` en el report, §8b del notebook).
