# Anatomía de los `<blockquote>` y decisión de diseño RAG

> Análisis del "aparato editorial" del BOE (todo lo que el BOE envuelve en `<blockquote>`) sobre las
> **20 normas descargadas**, y la decisión de qué conviene conservar dado que el RAG solo trabaja con
> **normas en vigor**. Datos de inspección directa del raw. Fecha: 2026-06-15.

## 1 · Qué es un `<blockquote>` aquí

El BOE mete **todo el aparato editorial** (lo que NO es el texto vigente del artículo) dentro de cajas
`<blockquote>`: notas de modificación, redacciones anteriores, avisos de vigencia, sentencias del TC…
El `class`/`caduca` de la caja es el "gancho" del visor web del BOE para decidir cómo/cuándo mostrarla.

**Total: 2445 blockquotes en las 20 normas.**

> Para ver en concreto qué se colaba o se perdía con el parser antiguo (antes y después sobre normas
> reales), ver la sección de evidencia de `01_parser_flujo_completo.md`.

## 2 · Las 6 familias (por su etiqueta de apertura)

| Etiqueta de apertura | Nº | Normas | Qué es |
|---|---|---|---|
| `<blockquote>` (sin clase) | 2091 | 20 | Notas a pie de modificación |
| `<blockquote class="soloTexto" caduca="AAAAMMDD">` | 266 | 14 | "Redacción anterior" (texto derogado) |
| `<blockquote class="noDesdeAAAAMMDD">` | ~71 | varias | Control de visibilidad por fecha |
| `<blockquote class="siempreSeVe">` | 11 | 6 | Avisos permanentes |
| `<blockquote class="nota_pie_2">` | 4 | 1 | Contenedor de notas (¡con atributo!) |
| `<blockquote class="docrel">` | 2 | 2 | Documento relacionado (jurisprudencia) |

**Ejemplos reales de cada una:**

**1) Sin clase → notas a pie** (86% de los casos):
```xml
<blockquote>
  <p class="nota_pie">Se modifica por el art. 3.1 del Real Decreto-ley 14/2019... Ref. BOE-A-2019-15790</p>
</blockquote>
```

**2) `soloTexto` + `caduca` → redacción anterior** (`caduca="20250403"` = deja de verse el 3-abr-2025):
```xml
<blockquote class="soloTexto" caduca="20250403">
  <p class="parrafo">Téngase en cuenta que esta actualización... entra en vigor el 3 de abril de 2025...</p>
  <p class="cita_con_pleca">Redacción anterior:</p>
  <p class="parrafo">"1. Los legítimos intereses económicos... (TEXTO VIEJO)"</p>
</blockquote>
```

**3) `noDesdeAAAAMMDD` → la fecha va en el nombre de la clase** ("no visible desde esa fecha"):
```xml
<blockquote class="noDesde20130101">
  <p class="parrafo">Atención: el apartado 13 surte efectos desde el 1 de enero de 2013.</p>
</blockquote>
<blockquote class="noDesde99999999">   <!-- 99999999 = fecha "infinita" = aviso permanente -->
  <p class="parrafo">Téngase en cuenta que se declaran inconstitucionales y nulos los incisos...</p>
</blockquote>
```

**4) `siempreSeVe` → avisos permanentes** (erratas, sentencias del TC, "Véase…"):
```xml
<blockquote class="siempreSeVe">
  <p class="parrafo">Téngase en cuenta que se declara la inconstitucionalidad del apartado 4...</p>
</blockquote>
```

**5) `nota_pie_2` → tiene atributo pero lleva NOTAS, no párrafos** (rompe la regla "atributo = párrafos"):
```xml
<blockquote class="nota_pie_2">
  <p class="nota_pie">Se modifica y se renumera por el art. 1.47 de la Ley Orgánica 8/2000... Ref. BOE-...</p>
  <p class="nota_pie_2">Su anterior numeración era art. 50.</p>
</blockquote>
```

**6) `docrel` → documento relacionado** (jurisprudencia):
```xml
<blockquote class="docrel">
  <p class="inforel">Información relacionada</p>
  <p class="nota_pie">Téngase en cuenta la Sentencia del TC 112/2018, de 17 de octubre...</p>
</blockquote>
```

**Dos ideas clave:** (a) `caduca="fecha"` (atributo) y `noDesdeAAAAMMDD` (fecha en la clase) son dos
mecanismos para "muéstralo/escóndelo en tal fecha"; para nosotros ambos = "no es el texto vigente".
(b) Tener atributo ≠ llevar párrafos (`nota_pie_2`/`docrel` llevan notas).

## 3 · Inventario completo (en las 20 normas)

**Clases de `<p>` dentro de blockquote:** `nota_pie` 4989 · `parrafo` 872 · `nota_pie_2` 792 ·
`cita_con_pleca` 170 (el marcador "Redacción anterior:") · `cita` 117 · `parrafo_2` 88 · `articulo` 23
(¡artículos enteros derogados!) · `cuerpo_tabla_centro/izq` 45 · `(sin-clase)` 15 · `capitulo_num/tit`
5 · `anexo_num/tit` 2 · `inforel` 2 · `cabeza_tabla` 1.

- **Sin atributos (tu "tipo 2"):** SOLO `nota_pie` (4979) + `nota_pie_2` (784). Confirmado 100%.
- **Con atributos (tu "tipo 1"):** mucho más que párrafos — citas, tablas, artículos/capítulos enteros
  derogados, e incluso algunas notas.

**Etiquetas inline dentro de los `<p>` (las que no se ven a simple vista):** `a` 5746 (enlaces a otras
normas, "Ref. BOE-…"; de aquí saca `_extract_norm_id` la procedencia) · `strong` 67 · `ins` 49
(texto insertado, marcado editorial) · `sub` 5 · y una tabla forma B (`table`/`tr`/`td`).

**Casos límite:** 0 blockquotes anidados · 1 con `<table>` · 0 con `<img>` · **30 `<p>` estructural/`articulo`
dentro de blockquote** (artículos/capítulos enteros derogados, sobre todo en Consumidores e Impuestos
Especiales) → los que dispara el warning "¿vigente mal envuelto?".

---

# Parte B · ¿Conviene quedarse con algo? (decisión de diseño RAG)

Contexto: el RAG **informa al ciudadano sobre la ley VIGENTE** (no asesoramiento vinculante). La
arquitectura da **tres destinos** para cada contenido:

1. **Cuerpo indexado** (`kept`→chunks→embeddings→retrieval): lo que se busca y se da al LLM. **Solo ley vigente.**
2. **Metadato unido por `block_id`** (`notes`/`history`, NO se embebe): disponible al responder, sin contaminar el retrieval.
3. **Descartado** (`dropped`).

Criterio técnico: meter texto editorial en el cuerpo mete **ruido léxico** (fechas, "Ref. BOE-…", texto
viejo casi-duplicado) que **degrada el retrieval** —y de forma asimétrica BM25 vs denso, justo lo que
mide el flagship—. El listón para indexar es altísimo.

## Veredicto por familia

| Familia | ¿Al cuerpo indexado? | ¿Metadato? | Por qué |
|---|---|---|---|
| Notas a pie (bare, nota_pie_2) | No | Sí (procedencia) | Quién y cuándo cambió la norma; útil para "¿desde cuándo?" y el aviso de actualización. Indexarlo añadiría ruido. |
| Redacción anterior (soloTexto, caduca) | No (peligroso) | No | Ley no vigente; redundante con el historial de versiones. Es el error que se corrigió. |
| Avisos de fecha de efecto (noDesde, "surte efectos desde…") | No | Sí (temporal) | Metadato temporal, como las notas a pie. |
| Inconstitucionalidad o interpretación del Tribunal Constitucional (siempreSeVe, noDesde99999999) | Caso especial | Sí, destacado | No es texto viejo: cualifica la validez del texto vigente. Ver abajo. |
| Documento relacionado (docrel) | No | Opcional | Punteros a jurisprudencia; secundario. |

**Respuesta directa:** para el **cuerpo indexado, NO interesa quedarse con ninguno**. El producto es el
texto vigente; lo editorial es ruido o provenance. Como metadato, las notas a pie sí valen (ya están en
`notes`). **Y hay una excepción real.**

## La excepción que SÍ importa: declaraciones del Tribunal Constitucional

Ejemplos reales del corpus MVP:
- LBRL (`BOE-A-1985-5392`): "se declaran **inconstitucionales y nulos** los incisos destacados…"
- `BOE-A-2015-11430`: "se declara la **inconstitucionalidad del apartado 4**…"
- Subvenciones (`BOE-A-2003-20977`): "Se declara la **constitucionalidad del apartado 2, siempre que se
  interprete** en los términos…"

**Esto NO es la redacción anterior. Cualifica el texto que SÍ está vigente:**
- "inconstitucional y nulo" = ese inciso **ya no es derecho válido**, aunque el texto siga apareciendo.
- "constitucional siempre que se interprete en el sentido X" = **límite interpretativo** obligatorio.

Para un RAG que informa de la ley vigente, **tirarlo en silencio es un fallo de corrección** (puede
responder con un inciso anulado o sin el matiz obligatorio). Es la única familia donde "dropear" no es
perder ruido.

**Cómo se usaría bien (encaja en la arquitectura):** NO como texto a embeber (seguiría siendo ruido),
sino como una **bandera de validez (`validity_caveat`) pegada al `parent` afectado** (metadato unido).
El generador la adjunta a la evidencia: *"Aviso: el TC declaró nulo el inciso X / debe interpretarse en
el sentido Y."*

## Veredicto final (con alcance)

1. **Para el MVP / el RAG actual, el comportamiento es CORRECTO:** no indexar ninguna blockquote en el
   cuerpo; notas a pie como metadato; el resto fuera. El cuerpo limpio es lo que da validez a la
   comparación de retrieval.
2. **La única mejora con valor real es la bandera de constitucionalidad/interpretación** (TC). Pero es
   **rara**, **jurídicamente delicada** (atribuir mal una nulidad es peor que omitirla) y **fuera del
   alcance del MVP** → **trabajo futuro**. El aviso legal estático mitiga parcialmente mientras tanto.
3. **No "enriquecer" el corpus con lo editorial:** casi todo degradaría el retrieval. La disciplina de
   *cuerpo = solo ley vigente* es una fortaleza del diseño.

> **TRABAJO FUTURO (post-MVP):** bandera de validez que surfacee las declaraciones del TC
> (inconstitucionalidad/nulidad/interpretación) sobre el bloque afectado, sin indexar el texto editorial.
