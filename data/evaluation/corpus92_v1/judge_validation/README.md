# Validación del juez LLM (OE-06) — corpus92, split dev

Evidencia de la validación del juez LLM (`gemma3:12b`) contra **anotación humana del autor** sobre las
respuestas de generación en dev (corpus92, k=3, generador `qwen2.5:7b`), re-anotadas sobre el **perfil
de consulta I1** (el ganador, OE-03).

## Ficheros

- `anotacion_corpus92_dev.jsonl` — **gold humano** (irremplazable): por `query_id`, `human_faithful`
  (L3) y `human_correctness` (L5). **31 respuestas** anotadas en fidelidad y en corrección; el acuerdo
  con el juez se computa sobre las **30** (fidelidad) y **15** (corrección) con veredicto del juez disponible.
- `anotacion_corpus92_dev_anotada.md` — fuente legible de la anotación (con notas), tal cual la
  rellenó el autor. Generada con `scripts/annotation_worksheet.py` (scaffold → md → jsonl).
- `judge_agreement_v2.json` — κ/AC1 del juez con el **prompt v2** (`prompts/`) vs el gold.
- `judge_agreement_v3.json` — κ/AC1 del juez con el **prompt v3 calibrado** (`prompts/judge_v3/`),
  re-juzgado sin regenerar (`scripts/rejudge_report.py`).

## Resultado (OE-06: el juez se valida; se halla insuficiente)

| Dimensión | v2 (κ / AC1) | v3 (κ / AC1) |
|---|---|---|
| L5 corrección (κ ponderado) | 0.21 / 0.60 | 0.40 / 0.67 |
| L3 fidelidad (κ nominal) | 0.44 / 0.83 | 0.36 / 0.78 |

- El juez **sobre-acredita** en ambos ejes (L5 colapsa `partial`→`correct`; L3 detecta solo **2/6**
  respuestas infieles).
- Las **mismas 4 alucinaciones** (q92_002, q92_021, q92_031, q92_063) burlan al juez con prompt laxo
  (v2) **y** escéptico (v3) → el fallo de fidelidad es de **capacidad del modelo**, no de prompt.
- κ < 0.6 en ambas dimensiones → **L3/L5 NO se citan como métrica de calidad del sistema.**

## Decisión (2026-06-28)

Se reportan **L3/L5 vía la anotación humana** (señal I1: fidelidad **81%** = 25/31; corrección **18
correct / 10 partial / 3 incorrect** de 31) y se **documenta la limitación** del juez-LLM. n pequeño
(κ sobre 15/30), IC anchos; cifras condicionales a que el sistema responda (las abstenciones no se anotan).

Reproducir: `scripts/validate_judge.py --report <report> --dataset-dir data/evaluation/corpus92_v1
--annotations data/evaluation/corpus92_v1/judge_validation/anotacion_corpus92_dev.jsonl`.
