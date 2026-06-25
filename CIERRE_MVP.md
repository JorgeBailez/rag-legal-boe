# CIERRE_MVP — plan ordenado y criterios de aceptación

> 🗄️ **HISTÓRICO (MVP corpus-10, cerrado).** El proyecto avanzó a **corpus-92** y la pata de
> recuperación (OE-01..OE-04) ya está cerrada (el denso gana el flagship). Estado vivo en
> `PROGRESO.md`; evidencia en `docs/decisiones_de_diseno.md`. Este doc se conserva como registro.

> **Qué es esto:** el plan para **cerrar el MVP con nota** — dejar la base (F1 pipeline + F2 índice
> denso + F3 generación + P0 evaluación) **validada y reproducible** sobre el corpus de **10 normas**.
> **Fuera de alcance** (es la fase siguiente, ver `PLAN.md`): comparación **BM25/denso/híbrido** y
> **ampliación del corpus a 100**.
>
> Evidencia actual (PRE-fix, quedará invalidada — ver §0): `RESULTADOS.md`. Estado vivo:
> `PROGRESO.md`. Contexto estable: `CLAUDE.md`. Creado 2026-06-14.
>
> Leyenda dueño: **🤖** asistente · **🧑** autor (validación jurídica/decisiones) · **🖥️** servidor
> `dslab01` (cómputo pesado, horas en CPU). `[ ]` pendiente · `[x]` hecho.

---

## ESTADO GLOBAL DEL CIERRE (2026-06-21) — LEER PRIMERO

Mapeo de fases (corpus 10) a hoy:

| Fase | Qué | Estado |
|---|---|---|
| A · parser reintegrado | fixes editorial+tablas en `cierre-mvp` (`28a853a`) | ✅ cerrada |
| B · corpus limpio (F1) | 10/10 reprocesadas; `audit_corpus.py` "Sin flags" | ✅ cerrada |
| C · índice denso (F2) | bundle `e5-large-instruct__j1__42105deb4afe` | ✅ cerrada |
| D1 · gold de **relevancia** (L1) | reviewed; Gate C dev✅/ooc✅/test 19-20 | ✅ cerrada |
| D2 · gold de **generación** (answer_keys) | **80/80 `reviewed`** (grounded, audit "Sin flags"); Gate C gen = test 19/20 (trampa q0051) | ✅ revisado (pendiente spot-check 🧑) |
| E · bake-off denso | 5 modelos (gte aplazado); §1 de `RESULTADOS.md` | ✅ cerrada (gte known-issue) |
| **FLAGSHIP** *(era fase siguiente, no del cierre)* | denso/BM25/híbrido + heading-boost+ley + fusión ponderada + test held-out | ✅ HECHO (§2 de `RESULTADOS.md`) |
| F · generación + juez + κ (F3+P0) | **NO corrida sobre el corpus limpio**; κ pre-fix=0.302 (<0.6, **void**); L3 n=3 | ❌ ABIERTA |
| G · higiene/cierre | `RESULTADOS.md` regenerado ✅; commit de cierre pendiente | ◑ parcial |

**Lo que falta para "MVP cerrado con nota":** **solo la Fase F** (correr generación + juez + κ sobre
el corpus limpio). **D2 (gold de generación) cerrado** (80/80 reviewed, pendiente spot-check del autor).
Todo lo demás está cerrado — y el flagship (que `PLAN.md` ponía como fase siguiente) ya está hecho y
validado en `dev` + `test`. Recordatorio de prioridad (`PLAN.md` §0/§5): **la generación es
SECUNDARIA** y no bloquea la entrega; las conclusiones fuertes van sobre **retrieval** (el flagship).

**Qué se puede CERRAR AHORA (corpus 10, sin esperar al 100):**
- 🤖🖥️ **Re-correr la generación sobre el corpus limpio** (`run_generation_eval.py` con el bundle
  nuevo). `--no-judge` da **L4 (citas) + L6 (abstención) + key-fact (L5 parcial) SÓLIDOS** sin
  depender del juez; con `--judge-model gemma3:12b` da L3/L5 **provisionales**. Reemplaza el report
  pre-fix (void) y recupera el hallazgo "seguro pero conservador" con cifras válidas.

**Qué necesita al AUTOR (🧑) — no se puede cerrar sin ti:**
- Promover `answer_keys` `draft → reviewed` (revisión jurídica) → cierra **D2**.
- Anotar κ (corrección + **fidelidad L3 n≥20**) sobre el report limpio → valida el juez (**F2/F3**).
  El juez (`gemma3:12b`) ≠ generador (`qwen2.5:7b`) ya está fijado; falta la κ sobre datos limpios.

**Qué necesita de verdad el CORPUS 100:** potencia estadística de las métricas de generación (más
preguntas) y un gold mayor. Las L3–L6 **se pueden calcular** en corpus 10 (es su alcance), pero las
conclusiones de generación quedan provisionales hasta tener κ≥0.6 + n suficiente.

**Recomendación:** cerrar ahora lo mecánico (re-correr generación sin/ con juez sobre el corpus
limpio y dejar el report como evidencia), marcar D2/κ como "requieren autor", y documentar que la
versión COMPLETA y con potencia de generación se hará sobre el corpus 100. Así el MVP queda cerrado
hasta donde es honesto sin el humano, y **nada queda sin trazar**.

---

## 0 · Principio rector — la cascada de invalidación (LEER PRIMERO)

Este cierre **arranca tocando el parser** (fixes de aparato editorial `<blockquote>` + tablas `<td>`
crudas). Eso **cambia el texto de los chunks**, y al cambiar los chunks **se invalida TODO lo medido
aguas abajo**. La cadena de dependencias es:

```
PARSER (fix editorial + tablas)
  └─► [B] reprocesar 10 normas  → parents/chunks NUEVOS
          └─► [C] re-indexar embeddings → bundle NUEVO (fingerprint nuevo)
                  └─► [E] re-correr bake-off de modelos → los números del 06-06 quedan VOID
          └─► [D] re-auditar grounding del gold → los block_id/parents pueden cambiar
                  └─► [F] re-generar respuestas → cambian con el corpus
                          └─► re-juzgar + re-validar κ → las anotaciones humanas pueden requerir revisión
```

**Regla de oro:** **ningún número de `RESULTADOS.md` sobrevive al cambio de parser.** El bake-off
(06-06), la generación, la κ del juez — todo se **rehace**, y **en este orden**. Hacerlo desordenado
(p. ej. validar el gold o el juez antes de limpiar el corpus) = **tirar el trabajo y repetirlo.**

> **Recomendación de flujo:** hacer el cierre en una rama nueva `cierre-mvp` partiendo de `main`
> (`c3a150c`), trayendo **solo el código** de los fixes del parser (no los datos ni el tooling de la
> ampliación). Así `main` sigue siendo el ancla limpia del MVP y el cierre es revisable.

---

## 1 · Análisis crítico del estado (por qué hace falta este cierre)

El MVP tiene **mucha ingeniería sólida**, pero sus **conclusiones descansan sobre cimientos sin
validar**. Honestamente, hoy **ningún número del MVP es plenamente defendible**:

1. **Corpus con bugs de datos (descubiertos esta sesión).** Las 10 normas incluyen **Haciendas
   Locales**, cuya **tabla de coeficientes de plusvalía (art. 107) está AUSENTE** del corpus (bug de
   tablas `<td>`), y varias normas base tienen **redacción derogada indexada como vigente** (leak
   editorial `<blockquote>`). → El índice, el retrieval y la generación se construyeron sobre datos
   defectuosos.
2. **Gold sin validar.** 70 preguntas / 62 judgments / 58-12 respondibles, **todo `draft`**,
   `gate_c_ready=false`. → Ninguna métrica de evaluación significa nada todavía.
3. **Juez sin validar.** Corrección κ=0.447 / ponderado 0.547 (**<0.6**), n=32, IC enorme; fidelidad
   **L3 con n=3 = sin medir**. → `faithfulness=0.992` y `correctness=0.891` no son defendibles.
4. **Bake-off incompleto.** `gte-multilingual-base` está **registrado pero nunca probado**; la
   afirmación "elegí el mejor modelo con evidencia" tiene un hueco.

**"Con nota" = convertir esos cuatro puntos de *provisional* a *validado*.** No es hacer más cosas;
es **cerrar lo que está abierto**, y dejar constancia escrita (esta checklist) para no volver a
perder el control.

---

## 2 · Alcance

| Dentro del cierre del MVP | Fuera (fase siguiente — `PLAN.md`) |
|---|---|
| F1: corpus de **10** normas limpio (parser corregido) | Comparación **BM25 / denso / híbrido** |
| F2: índice denso baseline (re-indexado) | Ampliación del corpus a **100** |
| Bake-off denso **completo** (incl. `gte-multilingual-base`) | Reranker, query-rewriting |
| F3: generación fundamentada | Banco de pruebas ampliado (~150–200) |
| P0: gold de 70 **validado** + juez con κ + métricas L1–L6 | |

**Opcionales** (suben nota; si se difieren, documentar): experimento `prompts/v2` (sobre-abstención),
arreglo de los 3 fallos de generación (q0001/q0013/q0038).

---

## 3 · Plan por fases (ORDEN OBLIGATORIO)

### Fase A · Reintegrar el parser (el disparador) — 🤖
- Crear rama `cierre-mvp` desde `c3a150c`.
- Traer **solo el código** de los fixes desde `respaldo-post-mvp`: `src/boe/parser.py`,
  `src/quality/corpus_audit.py`, `scripts/audit_corpus.py`, y los tests `test_parser_editorial.py`,
  `test_corpus_audit_editorial.py`, `test_parser_tables.py`, `test_corpus_audit_tables.py`.
  **No** traer manifests/datos ni el tooling de ampliación (recon/`--only-new`) salvo que se decida.
- **Criterios de aceptación:**
  - [ ] El código de los dos fixes + sus tests está en la rama; el corpus sigue siendo **10 normas**.
  - [ ] `uv run ruff check .` limpio (incluye `.ipynb`) y `uv run ruff format --check .` limpio.
  - [ ] `uv run pytest --deselect tests/test_encoder.py::test_revision_unpinned_blocks_without_flag` en verde.
  - [ ] `uv run python -m src.contracts.export_schemas --check` sin drift.

### Fase B · Corpus limpio (F1) — 🤖 código · 🖥️ corrida

> **ESTADO (2026-06-16): Fase B completada en LOCAL.** 10/10 reprocesadas; tablas A y B linealizadas
> por fila (a72/a86/a107/a124 con concepto↔valor emparejados); `audit_corpus.py --strict` →
> `pre_embedding_readiness.ready=True`, `blocking_findings: []` (editorial_leak/note_leak/
> table_cell_dropped = 0; H4 descartado, H5 confirmado). **`raw_integrity` resuelto**: manifests de las
> 4 normas con deriva del BOE re-baselinados desde disco (recompute sha256/size + `downloaded_at`
> refrescado al mtime real), 10/10 OK. Restos no bloqueantes: `H3_oversized` (deferred) + 2 WARN de
> "Redacción anterior" derogada legítima. **Pendiente:** suite completa (torch) y reindex en servidor (Fase C).

- Reprocesar las 10 normas: `parse → chunk → audit` sobre el corpus del MVP.
- **Linealización de tablas forma A: YA IMPLEMENTADA en el parser** (2026-06-15). Todas las tablas (A y
  B) se linealizan por fila (`concepto | valor`); los `<p>` dentro de `<table>` se saltan (no duplican).
  Entra automáticamente en este reproceso → no añade reindex. Verificado: subconjunto de tests verde (92)
  + prueba real sobre Haciendas Locales (a72/a86/a124 con valores emparejados). Contrato intacto. Análisis
  en `docs/analisis/04_tablas.md`. **Criterio de aceptación extra de Fase B:** a72/a86/a95/a124 muestran
  filas `cuerpo_tabla_fila` emparejadas en `data/processed/parents/BOE-A-2004-4214.json`.
- **Criterios de aceptación:**
  - [ ] `audit_corpus.py --strict` → **"Sin flags"** (o solo known-issues documentados y aceptados).
  - [ ] `editorial_leak = 0`, `table_cell_dropped = 0`, `note_leak = 0`.
  - [ ] Verificación puntual: la **tabla de plusvalía (a107 Haciendas Locales)** y demás tablas
        vigentes **presentes** en parents/chunks; "Téngase…"/"Redacción anterior:" **ausentes** del cuerpo.
  - [ ] Cuarentena correcta donde toque; `pre_embedding_readiness` OK.

### Fase C · Índice denso (F2) — 🖥️
- Re-generar el bundle baseline (`e5-large-instruct · J1 · I2_CITIZEN_LEGISLATION`) sobre el corpus limpio.
- **Criterios de aceptación:**
  - [ ] `validate_dense_index.py` OK; bundle inmutable nuevo publicado (fingerprint nuevo).
  - [ ] `CLAUDE.md` actualizado con el **nuevo checkpoint**; bundle viejo marcado obsoleto.

### Fase D · Gold validado (P0) — 🤖 propone · 🧑 valida
> **ESTADO (2026-06-19): COMPLETADA (gold de RELEVANCIA).** Gold revisado con relevancia graduada
> 2/1/0, evidencia por párrafo, multi_parent, negativos tentadores y trampas; pooling multi-sistema
> (`scripts/build_eval_candidates.py`). `audit_eval_dataset.py` "Sin flags". Gate C checkpoint: dev ✅
> (49≥40) · ooc ✅ (10/10) · test **19/20** (aceptado: q0051 es trampa temporal, rel 0). El gold de
> GENERACIÓN (`answer_keys`) sigue `draft` (2ª pasada, secundario). Pendiente: spot-check del autor.
*Depende de B (el grounding se comprueba contra el corpus limpio).*
- **D1 · Relevancia (L1):** re-grounded los `judgments` contra el corpus limpio (los `block_id` pueden
  haber cambiado), relevancia 1/0, `paragraph_orders` reales.
- **D2 · Generación:** `reference_answers` grounded (cada hecho en el texto citado), `key_facts`
  literales, `multi_parent` marcado donde sea legítimo.
- Promoción `draft → reviewed` según política de `PLAN.md` (🤖 marca lo 100% seguro; 🧑 revisa lo
  dudoso + muestreo de lo `reviewed`).
- **Criterios de aceptación:**
  - [ ] `audit_eval_dataset.py` → **"Sin flags"** contra el corpus limpio.
  - [ ] El subconjunto usado en evaluación está **`reviewed`** y auditado; **Gate C abre** para él.
  - [ ] Reparto por área/dificultad documentado; sin fuga dev/test.

### Fase E · Bake-off de modelos COMPLETO (F2) — 🖥️
> **ESTADO (2026-06-19): bake-off denso HECHO** (`bench_20260619T063600Z`, 5 modelos, dev n=50;
> gte aplazado). Resultados y conclusiones en `RESULTADOS.md`. Gana **e5-large-instruct·J1·I2** (IC
> solapan; solo e5-base claramente peor). Hallazgo: el denso falla **directa_articulo** (~0.35) →
> motiva el híbrido. Abstención por umbral OK (AUC ~0.99). Pareto: bge-m3/qwen3 dominados en CPU.
> Pendiente: cifra held-out en `--split test`.
*Depende de B (corpus) y D1 (gold de relevancia).*
- **E1 · Smoke** de `gte-multilingual-base` (viabilidad CPU: encode/latencia/RAM, revisión fijada por hash).
  > **ESTADO (2026-06-18): gte APLAZADO.** Pinneado (`9bbca17d`) y código remoto revisado, pero su
  > `modeling.py` (transformers 4.39) es **incompatible con el transformers del entorno** (el que exige
  > `qwen3`) → `IndexError` en el RoPE al encodear. Se cierra el MVP con **e5×3 + bge-m3 (+ qwen3)** y se
  > deja gte para un venv con transformers pinneado. `qwen3-0.6b` se incluye por calidad pese a ser
  > **lento en CPU** (~0,1–0,2 docs/s → coste prohibitivo, hallazgo). Detalle en `docs/known_issues.md`.
- **E2 · Benchmark** sobre el **corpus limpio**: re-correr los 5 modelos del bake-off + `gte`
  (calidad × perfil de query; chunking si aplica), con IC bootstrap y diferencia pareada vs baseline.
  > **Maquinaria ampliada (2026-06-17, ver `docs/analisis/06_diseno_experimental.md`):** el bake-off
  > ahora produce además **cortes por `query_style`/`difficulty`**, **bootstrap pareado de cada modelo
  > vs el baseline** (`--baseline-alias`), **experimento de abstención** sobre `out_of_corpus`
  > (ROC-AUC + umbral; `--skip-abstention` para omitir) y **frontera calidad/coste** (Pareto). Todo
  > como secciones nuevas del `report.json`; solo separan modelos cuando el gold esté enriquecido (D).
- **E3 · Decisión de modelo** documentada; si `gte` u otro **supera** a `e5-large-instruct`, cambiar
  checkpoint → **volver a Fase C** con el nuevo modelo.
- **Criterios de aceptación:**
  - [ ] Report de benchmark con los **6 modelos** sobre corpus limpio + gold validado.
  - [ ] Ganador documentado con evidencia estadística (CIs, bootstrap pareado).
  - [ ] Checkpoint confirmado o actualizado en `CLAUDE.md`.

### Fase F · Generación + juez + κ (F3 + P0) — 🖥️ (horas) · 🧑 anota
*Depende de B (corpus), C (bundle) y D2 (answer_keys).*
- **F1 · Re-generar** la evaluación de generación **sin juez** sobre datos limpios (abstención, citas, key-fact).
- **F2 · Re-generar con juez** y **re-validar κ**: re-anotar las filas cuya **respuesta haya cambiado**
  respecto a las 32 originales (el corpus limpio cambia respuestas, sobre todo en Haciendas Locales y
  normas con leak); **ampliar la anotación de fidelidad L3 a n≥20** (hoy n=3).
- **F3 · Decisión sobre el juez:** κ≥0.6 → L3/L5 fiables; si sigue <0.6 → juez más fuerte
  (`gemma3:27b`) **o** reportar L3/L5 como provisionales con caveats explícitos.
- **Criterios de aceptación:**
  - [ ] Report de generación sobre **corpus + bundle + gold limpios**.
  - [ ] `judge_agreement.json` con **κ corrección ≥ 0.6** (o decisión documentada) **y L3 con n≥20** interpretable.
  - [ ] Métricas L1–L6 reportadas **con sus caveats** (p. ej. `citation_precision` no se reporta sola).

### Fase G · Higiene y cierre — 🤖 + 🧑
- **Criterios de aceptación:**
  - [ ] Gates verdes y reproducibles (`ruff` + `pytest` deselec. + `export_schemas --check`).
  - [ ] `RESULTADOS.md` **regenerado** con los números POST-fix (los actuales quedan marcados como void).
  - [ ] `PROGRESO.md`, `known_issues.md` y `CLAUDE.md` (checkpoint) sincronizados con el estado real.
  - [ ] Esta checklist completada; **commit de cierre del MVP**.

---

## 4 · Checklist maestra — "MVP cerrado con nota"

- [x] **Corpus limpio:** sin leak editorial, sin tablas perdidas; audit "Sin flags" sobre las 10.
- [x] **Índice denso re-generado** sobre el corpus limpio; checkpoint actualizado (`…42105deb4afe`).
- [x] **Bake-off** sobre datos limpios; modelo elegido con evidencia. *(5/6: `gte` aplazado como
  known-issue documentado, no 6 — aceptado.)*
- [x] **Gold validado:** **relevancia (L1) ✅** y **generación (answer_keys) ✅ 80/80 reviewed**
  (audit "Sin flags"); Gate C = test 19/20 (trampa q0051, aceptado). Pendiente spot-check del autor.
- [ ] **Juez validado:** κ corrección ≥0.6 y L3 n≥20 sobre el **corpus limpio** — **pendiente** (el
  κ=0.302 pre-fix es void; requiere re-correr generación + anotación 🧑).
- [ ] **Generación re-evaluada** sobre datos limpios; L1–L6 con caveats — **pendiente** (Fase F).
- [x] **Gates verdes** (ruff + 619 pytest + schemas) **+ `RESULTADOS.md` regenerado** (incl. flagship).
- [ ] **Commit de cierre** y decisión consciente de qué queda como known-issue.

> **Resumen:** cierre del MVP **~75 %**. Bloque de retrieval (A–E) + flagship: **cerrados**. Bloque de
> generación (D2 + F): **abierto**; lo mecánico se cierra ahora (re-correr generación limpia), lo
> jurídico (gold reviewed) y la κ requieren al autor; la versión con potencia, al corpus 100.

---

## 5 · Lo que NO se toca en esta fase

- Comparación **BM25 / denso / híbrido** (flagship) → fase siguiente.
- **Ampliación del corpus** (10 → 100) → fase siguiente.
- Reranker, query-rewriting, banco de pruebas ampliado → fase siguiente.

> Cuando esta checklist esté al 100%, el MVP queda **cerrado y validado**, y el flagship se construye
> sobre terreno firme — que es exactamente lo que faltaba para no volver a perder el control.
