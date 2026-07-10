# Tipos de `<bloque>` y su peso en el retrieval

> Análisis del atributo `tipo` de los `<bloque>` del BOE (`<bloque id="…" tipo="precepto" titulo="…">`)
> sobre las 20 normas, cómo los trata el parser, su importancia para retrieval, y una **evaluación
> crítica** de si las posibles mejoras merecen la pena. Datos de inspección directa. Fecha: 2026-06-15.

## 1 · Los 5 tipos que existen (20 normas)

| `tipo` | Nº | Qué es | Tratamiento del parser | Peso en retrieval |
|---|---|---|---|---|
| `precepto` | 3029 | La ley operativa: artículos y disposiciones | 3019 indexables, 10 no (rótulos sin cuerpo) | máximo (~99 %) |
| `encabezado` | 741 | Rótulos estructurales y cabeceras de anexo | 718 estructurales (no indexables), 20 anexos y 3 con cuerpo (indexables) | mixto (ver sección 3) |
| `preambulo` | 20 | Exposición de motivos | 20 indexables | medio-bajo (contexto, no ley) |
| `firma` | 20 | Firma y fecha | no indexable | nulo |
| `nota_inicial` | 9 | Nota editorial inicial del consolidado | no indexable | nulo |

## 2 · `precepto` — el corazón (desglose por título)

| Subtipo | Nº | Qué dice | Consulta del ciudadano |
|---|---|---|---|
| **Artículo** | 2412 | Reglas sustantivas ("plazo para…", "derecho a…") | altísima |
| **Disposición adicional** | 313 | Reglas complementarias, regímenes especiales | media-alta |
| **Disposición final** | 151 | Entrada en vigor, habilitaciones, modificaciones a otras normas | baja por ciudadanos; clave para fechas |
| **Disposición transitoria** | 123 | Régimen de las situaciones en curso durante el cambio | media en casos concretos |
| **Disposición derogatoria** | 19 | Qué deroga la norma | baja |
| **(rótulo estructural disfrazado)** | 10 | "Sección 3.ª…" tipada como precepto, **sin cuerpo** | nulo → no indexable |

→ Los **artículos** son el pan del RAG; las **disposiciones** son ley igual, pero más "de fontanería".
Los 10 rótulos disfrazados los caza el parser **por no tener cuerpo** (no por el tipo).

## 3 · `encabezado` — el tipo de doble cara (741)

| Sub-rol | Nº | Qué es | Indexable | Peso |
|---|---|---|---|---|
| structural_heading | 718 | "TÍTULO I", "CAPÍTULO II", "SECCIÓN 3.ª" | No | nulo directo, alto indirecto (jerarquía, cita, filtros) |
| annex | 20 | Cabeceras de anexo con cuerpo (tablas, listas) | Sí | alto (plusvalía, puntos…) |
| content_heading | 3 | Cabeceras con cuerpo | Sí | medio |

## 4 · `preambulo` — el indexable "blando" (20)

Exposición de motivos: por qué y para qué se hizo la ley. **Se indexa** (útil para "¿cuál es el
objetivo de esta ley?"), **pero NO es ley operativa.** Riesgo sutil: responder una pregunta operativa
("¿qué plazo tengo?") desde una justificación narrativa. El parser ya lo etiqueta
`semantic_role="preamble"` → la señal para tratarlo distinto **ya existe**.

## 5 · `firma` y `nota_inicial` — fuera, y bien (29)

- `firma`: "Madrid, … El Rey … El Presidente del Gobierno …". Ceremonial → nulo, excluido.
- `nota_inicial`: nota del editor sobre el consolidado ("Incluye la corrección de errores…") →
  provenance, no ley, excluido.

## 6 · El principio que une todo

**El parser NO se fía del `tipo`; mira si hay cuerpo recuperable** (`has_retrievable_body`). Por eso
**10 "precepto" quedan FUERA** (rótulos sin cuerpo) y **23 "encabezado" entran** (anexos con cuerpo).
Misma filosofía que con los blockquote: decide el **contenido/contenedor**, no la etiqueta nominal.

## 7 · Ranking de importancia para retrieval

```
Máximo  precepto/artículo ........ las reglas; la respuesta a casi todo
Alto    precepto/disposiciones ... ley operativa (adicional > transitoria > final > derogatoria)
Alto    encabezado/annex ......... contenido de los anexos (tablas)
Medio   encabezado/structural .... nulo directo, imprescindible para jerarquía, cita y filtros
Bajo    preambulo ................ contexto y objetivo, no ley: indexable pero secundario
Nulo    firma, nota_inicial ...... aparato editorial; queda fuera (correcto)
```

---

# Parte B · Conclusiones y evaluación crítica de mejoras

## 8 · Lo que este análisis CONFIRMA (a diferencia del de blockquotes)

Este análisis **no destapa bugs de corrección** — valida que el parser ya hace lo correcto:
- Indexabilidad de los 5 tipos: correcta.
- "El cuerpo decide, no el `tipo`": aplicado (10 disfrazados fuera, 23 anexos dentro).
- La jerarquía de secciones funciona en el caso normal: **182 de 197 rótulos de sección vienen como
  `encabezado`** → 1124 bloques tienen `section`, en 11 de 20 normas.
- Las subsecciones internas del preámbulo (48) **no contaminan** la jerarquía (acierto, no olvido).

→ **~95 % de lo descubierto ya estaba aplicado y bien.**

## 9 · Mejoras candidatas — esfuerzo vs beneficio (evaluación crítica)

| Mejora | Dónde | Esfuerzo | Beneficio | ¿Significativo? | Veredicto |
|---|---|---|---|---|---|
| **Jerarquía desde rótulos disfrazados** (15 secciones tipo `precepto`) | parser (`_update_hierarchy`) | bajo-medio: código + test + **reproceso** (cambia `hierarchy.section` → artefactos) | bajo: solo `hierarchy.section` de unos pocos artículos; **NO afecta a la cita**; ~0 al flagship | **No** | **Diferir** |
| **Down-weight del preámbulo** | downstream (retrieval/generación) | medio: tuning + evaluación en el gold | incierto: riesgo **teórico, no observado** (el denso recupera bien: 0 artículos correctos no recuperados) | **No demostrado** | **Diferir** (guiar por evidencia, no especular; es fase siguiente) |
| **Sub-rol de disposiciones** (distinguir de artículos) | parser/contrato | bajo, pero toca el contrato (`export_schemas` + anti-drift) | nulo: la distinción **ya está en `full_title`** | **No** | **Descartar** (gold-plating) |

### Razonamiento detallado

**1) Jerarquía desde rótulos disfrazados.** Real pero **menor**. `_update_hierarchy` solo dispara con
`encabezado`; las 15 "Sección X" tipadas `precepto` no actualizan la jerarquía → los artículos bajo
ellas tienen `section` vacía o heredada (stale). PERO: (a) **no afecta a la cita** (`citation_label`
no usa la sección); (b) solo toca `hierarchy.section` → filtros/contexto del chunk; (c) **forzaría un
reproceso** (y posiblemente reindex) por un beneficio casi nulo, y (d) **cero impacto en el flagship**
(la comparación de retrieval no depende de la sección). → No merece la pena ahora. El arreglo, si se
hiciera: que la jerarquía dispare también desde bloques **no indexables cuyo único contenido es un
rótulo estructural**, sea cual sea su `tipo`.

**2) Down-weight del preámbulo.** Es la idea **más interesante conceptualmente**, pero **NO es del
parser** (la señal `semantic_role="preamble"` ya se produce). Sería tuning de retrieval/generación. Y
sobre todo: **no hay evidencia de que el preámbulo cause respuestas malas** — el retrieval denso
recupera bien el artículo correcto. Tunear contra un riesgo teórico es optimización prematura. →
Diferir hasta que la evaluación muestre un caso real de interferencia del preámbulo. Es fase siguiente.

**3) Sub-rol de disposiciones.** La información (artículo vs disposición adicional/transitoria/…) **ya
está en el título**, disponible para filtrar/pesar. Añadir un sub-rol al contrato dispararía regen de
schemas + anti-drift por valor nulo. → Descartar.

## 10 · Veredicto

```
Bugs de corrección descubiertos aquí ......... NINGUNO (esto valida el parser)
Mejoras que merecen la pena AHORA ............ NINGUNA
  · jerarquía rótulos disfrazados ............ real pero menor → diferir (pulido)
  · down-weight preámbulo .................... downstream + sin evidencia → diferir (fase siguiente)
  · sub-rol disposiciones .................... gold-plating → descartar
```

A diferencia del análisis de blockquotes/tablas (que destapó fugas de corrección y motivó un fix), el
de tipos de bloque **confirma el diseño**. La disciplina de "el cuerpo decide, no el `tipo`" ya cubre
los casos raros. **No hay nada que aplicar con urgencia; las mejoras candidatas o son insignificantes
o son trabajo futuro guiado por evidencia.**
