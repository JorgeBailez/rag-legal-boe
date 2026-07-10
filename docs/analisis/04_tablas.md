# Tablas: dónde están, cuáles indexar y la mejora pendiente

> Análisis de las tablas del BOE en las 20 normas: dónde aparecen, cómo se representan (forma A vs B),
> qué se indexa y qué no, y una **evaluación crítica** de la única mejora de preprocesado con valor real
> que ha salido de estos análisis. Datos de inspección directa. Fecha: 2026-06-15.

## 1 · Dónde están las tablas (raras pero de alto valor)

Solo **6 de las 20 normas** tienen tablas vigentes, y **no solo en anexos** — también en artículos y
disposiciones transitorias:

| Norma | Bloques con tabla vigente |
|---|---|
| BOE-A-1985-5392 (LBRL) | Artículo 75 bis |
| BOE-A-1992-28741 (Impuestos Especiales) | Artículo 70 · Disp. transitoria 7ª |
| BOE-A-1994-26003 | Disp. transitoria 2ª · 3ª |
| **BOE-A-2004-4214 (Haciendas Locales)** | Art. 72, 86, 95, 107, 124 |
| BOE-A-2015-11722 (Tráfico) | ANEXO II, IV, V |
| BOE-A-2017-12902 | ANEXO I, II, IV |

Contenido: **coeficientes, recargos, ponderación, tarifas, puntos** — justo los números que el
ciudadano pregunta de forma concreta. Tablas escasas, pero de las consultas más "de dato exacto".

## 2 · Cómo se representan (dos formas) y dónde caen

**Inventario de `<table>` (20 normas):**

| Forma | Fuera de blockquote (vigente) | Dentro de blockquote (editorial) |
|---|---|---|
| **A** — celdas en `<td><p class="cuerpo_tabla_*">` | 46 | 1 |
| **B** — texto crudo en `<td>` (sin `<p>`) | 7 | 0 |

- Las celdas `cuerpo_tabla_*`/`cabeza_tabla` (1322 vigentes + 46 editoriales) **siempre están dentro de
  un `<table>`** (no hay celdas "sueltas").
- Las **forma B** (7 tablas) son las que el parser `<p>`-céntrico perdía: `a107` (plusvalía) y `anii`
  (puntos) son las más notables.

## 3 · Decisión indexar / no indexar (RESUELTA y aplicada)

Misma regla que todo el aparato editorial: **el `<blockquote>` decide.**

| Caso | Decisión | Estado |
|---|---|---|
| Tabla **vigente** (fuera de blockquote) | **Indexar** (forma A vía `<p>`; forma B linealizada por fila) | aplicado |
| Tabla **editorial/derogada** (dentro de blockquote) | **No indexar** (a `dropped`) | aplicado |

Solo **1 tabla editorial** en 20 normas → el caso "no indexar" es marginal. La decisión está cerrada.

## 4 · El hallazgo real: forma A se guarda CELDA A CELDA (mala para responder)

El fix de tablas resolvió las **forma B** y las linealiza **por fila**; pero las **forma A** (las 46, la
mayoría) siguen guardándose **celda a celda** → el emparejado concepto↔valor se rompe.

**`a72` Haciendas (forma A) — recargo inmuebles desocupados (3 columnas):**
```
[cabeza_tabla]        Puntos porcentuales
[cabeza_tabla]        Bienes urbanos
[cabeza_tabla]        Bienes rústicos
[cuerpo_tabla_izq]    A) Municipios que sean capital de provincia o comunidad autónoma
[cuerpo_tabla_centro] 0,07
[cuerpo_tabla_centro] 0,06
```
→ Concepto ("Municipios capital") y valores (0,07 urbanos / 0,06 rústicos) en **párrafos separados**.
**No se sabe qué valor va con qué columna** → el LLM puede **emparejar mal** al responder
"¿recargo de bienes rústicos en municipio capital?".

**`a107` Haciendas (forma B, el fix) — plusvalía:**
```
[cabeza_tabla]      Periodo de generación | Coeficiente
[cuerpo_tabla_fila] Inferior a 1 año. | 0,15
[cuerpo_tabla_fila] 1 año. | 0,15
```
→ Concepto↔valor **emparejados**, sin ambigüedad.

**Conclusión incómoda:** el fix dejó las **forma B mejor representadas que las forma A**, y las forma A
son la mayoría (incluidos los recargos/coeficientes de Haciendas Locales — norma del MVP — y los anexos
de Tráfico).

## 5 · Evaluación crítica: ¿merece la pena arreglarlo?

**Mejora:** linealizar también las tablas **forma A por fila** (misma lógica `_linearize_table_rows`:
lee el texto de `<td>`/`<th>` —vía `itertext`, captura el `<p>` interno— y une la fila con ` | `;
suprimir la captura celda-a-celda de los `<p class="cuerpo_tabla">` dentro de `<table>`).

| Eje | Valoración |
|---|---|
| **Esfuerzo** | **Medio**: cambio acotado en `classify_version_paragraphs` + tests + reproceso + reindex |
| **Beneficio** | **Alto** para preguntas de dato exacto (recargos, coeficientes, tarifas, puntos): elimina la **ambigüedad de emparejado** que hoy puede dar respuestas mal pareadas |
| **Prioridad** | Ayuda a la **generación** (emparejar el valor), **no al flagship** (retrieval: el chunk se recupera igual). Generación es secundaria en `PLAN.md` |
| **A favor de hacerlo ya** | **Va de la mano del reproceso/reindex** que el cierre del MVP ya necesita → coste marginal; y elimina la inconsistencia A/B |

**Veredicto (distinto al de tipos de bloque):** **SÍ merece la pena**, pero como **ride-along del
reproceso/reindex del cierre del MVP**, no como tarea aparte. Es la **única mejora de preprocesado con
valor real** salida de estos análisis (las de tipos de bloque eran insignificantes). Si se quiere ceñir
estrictamente al camino crítico del flagship, se puede diferir (es calidad de *generación*, no de
*retrieval*) — pero estando ya tocando el parser y reindexando, **es el momento barato de hacerlo bien**.

> Mejora candidata (trabajo futuro): linealizar las tablas forma A por fila, para que todas las tablas
> (A y B) queden con el emparejado concepto-valor.

## 6 · Resumen

```
¿Dónde? ............ 6/20 normas, en artículos + disp. transitorias + anexos (fiscal/tráfico)
¿Cuáles indexar? ... vigentes (fuera de blockquote) = SÍ · editoriales (dentro) = NO  → RESUELTO/aplicado
Mejora pendiente ... linealizar forma A por fila (hoy celda-a-celda) → calidad de generación
  · esfuerzo medio · beneficio alto en datos exactos · ride-along del reindex del cierre → recomendado
```
