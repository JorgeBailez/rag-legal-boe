# El parser del BOE, explicado desde cero

> Documento de onboarding. Explica el flujo completo del parser para alguien que nunca ha visto XML.
> Código fuente: `src/boe/parser.py`. Ejemplos reales del corpus MVP. Fecha: 2026-06-15.

## 0 · El objetivo en una frase

El BOE da el texto de las leyes en ficheros de máquina enrevesados (XML) con versiones antiguas,
notas de edición y formatos raros. **El parser es el traductor** que los convierte en texto limpio,
estructurado y "solo lo que es ley HOY", listo para un buscador + IA.

> Entra: el volcado crudo de una norma del BOE. Sale: tres ficheros limpios y estructurados.

## 1 · Qué es un XML (para principiantes)

Información en **cajas etiquetadas dentro de otras cajas**: `<etiqueta>contenido</etiqueta>`.
- **Etiqueta (tag):** el nombre de la caja (`<parrafo>`).
- **Atributos:** "pegatinas" de la caja (`id="a19"`, `class="parrafo"`).
- Es un **árbol** (padres/hijos), como carpetas dentro de carpetas. "Parsear" = recorrer el árbol y
  sacar lo que interesa.

## 2 · Qué se descarga del BOE (la ENTRADA)

`scripts/download_boe_raw.py` → `src/boe/client.py` llama a la API pública del BOE con un id
(`BOE-A-2015-10565`) y guarda los XML en `data/raw/boe/<id>/` (raw = crudo, intocable) + un *manifest*
con sha256/tamaño. Cuatro ficheros (el parser ignora `full.xml` y `metadata_eli.xml`):

| Fichero | Qué es |
|---|---|
| `metadatos.xml` | DNI de la norma: título, rango ("Ley 39/2015"), fechas, URL oficial, estado de consolidación. |
| `analisis.xml` | Materias, notas y referencias a otras normas. **Opcional.** |
| `indice.xml` | Tabla de contenidos: bloques EN ORDEN + **`fecha_actualizacion` de cada bloque** (clave de vigencia). |
| `texto.xml` | El texto: bloques, cada uno con sus **versiones** en el tiempo, cada versión con párrafos/tablas. |

**Conceptos:**
- **Texto consolidado:** el BOE mantiene la ley actualizada; un artículo puede tener **varias versiones**.
- **Bloque (`bloque`):** una pieza (artículo, anexo, cabecera, preámbulo, firma). Tiene `id` y `tipo`.
- **Versión (`version`):** una redacción del bloque, con `fecha_publicacion`.
- **Vigencia:** cuál versión es la ley **ahora**.

## 3 · Qué produce (la SALIDA): tres contratos v2

Tres ficheros, **un único dueño por dato** (no se duplica):

| Salida | Rol | Analogía |
|---|---|---|
| `document_v2` | Mapa/descriptor: bloques con flags (indexable, cita, jerarquía). **Sin texto pesado.** | Índice del libro anotado. |
| `parents_v2` | **Dueño del texto vigente:** los párrafos reales de cada bloque. | El libro (solo páginas vigentes). |
| `history_v2` | Máquina del tiempo: versiones, notas de modificación, resolución de vigencia. | Registro de cambios. |

Se unen por `block_id`/`parent_id`. El **texto** solo vive en `parents`.

## 4 · El flujo en 11 fases

1. **Abrir y validar** (`load_xml` + `validate_response`): el BOE envuelve todo en
   `<response><status><code>200</code><data>…`; se comprueba el 200 y se saca `<data>`.
2. **Metadatos** (`parse_metadata`): título, `short_title` (rango+número → "Ley 39/2015"), URL, fechas
   (normaliza `20151001`→`2015-10-01`), estado de consolidación.
3. **Análisis** (`parse_analysis`): materias, notas, referencias. Opcional.
4. **Índice** (`parse_index`): lista ordenada de bloques + su `fecha_actualizacion` (criterio de vigencia).
5. **Texto + vigencia** (`parse_text_blocks` + `resolve_current_version`): ver §5 detallado.
6. **Clasificar el contenido** (`classify_version_paragraphs`): kept / notes / dropped. **Ver Parte B.**
7. **Semántica y flags** (`_block_semantics`, `_full_title`): rol, cuerpo recuperable, anexo, tabla… **Parte B.**
8. **Jerarquía** (`_update_hierarchy`): Libro/Título/Capítulo/… **Parte B.**
9. **Indexabilidad + cita** (`build_block_descriptor_fields`). **Parte B.**
10. **Ensamblar** los 3 contratos (`_build_document_v2`, `_build_history`, `_build_parents`) + validar Pydantic.
11. **Guardar** (`save_processed_bundle`): 3 JSON en `data/processed/{documents,histories,parents}/`.

## 5 · La regla de oro: resolución de vigencia

`resolve_current_version` decide qué versión es la ley hoy, **estricto y sin fallback**:

> La versión vigente es la que tiene una `fecha_publicacion` que coincide **EXACTAMENTE** con la
> `fecha_actualizacion` del índice **y** es la **máxima** fecha. Si encaja exactamente una → `resolved`.
> Si no (no coincide / coinciden varias / falta fecha / no es la máxima) → **CUARENTENA**.

- **El orden en el XML NUNCA decide la vigencia.**
- **Cuarentena** = el bloque no recibe texto vigente, no se indexa, no genera chunks, y queda marcado
  (estados: `missing_index_date`, `unresolved`, `ambiguous`, `index_not_max`, `invalid_date`).
- *Por qué tan estricto:* en derecho, mostrar una versión equivocada es peligroso → mejor abstenerse.

---

# Parte B · Profundización en las fases 6–9 (con ejemplos reales del MVP)

Normas de ejemplo: **Ley 39/2015 LPAC** (`BOE-A-2015-10565`, muy estructurada) y **Haciendas Locales**
(`BOE-A-2004-4214`, tablas).

## FASE 6 — `classify_version_paragraphs`: tres cubos

Recorre la versión vigente en **orden de documento** y reparte cada `<p>`/`<table>`:
1. ¿clase `nota_pie`/`nota_pie_2`, **o** texto empieza por "Téngase en cuenta"? → **notes**.
2. Si no, ¿está dentro de un `<blockquote>`? → **dropped**.
3. Si no → **kept** (cuerpo vigente).

> Regla de oro: se decide por **CONTENEDOR** (`<blockquote>`), no por la clase del párrafo. Dentro de
> un blockquote hay párrafos con la misma `class="parrafo"` que el vigente, pero son la versión vieja.

**Ejemplo real — kept + notes (Artículo 9 de LPAC):**
- FUERA del blockquote → `kept` (la ley de hoy):
  ```
  [articulo]  Artículo 9. Sistemas de identificación de los interesados...
  [parrafo]   1. Las Administraciones Públicas están obligadas a verificar la identidad...
  [parrafo_2] a) Sistemas basados en certificados electrónicos cualificados de firma...
  ```
- DENTRO del blockquote → `notes` (provenance, NO se indexa):
  ```
  [nota_pie]   Se modifica la letra c) del apartado 2 por la disposición final 1.1 de la Ley 11/2022... Ref. BOE...
  [nota_pie]   Se modifica por el art. 3.1 del Real Decreto-ley 14/2019... Ref. BOE-A-2019-15790#a3
  [nota_pie_2] Téngase en cuenta para su aplicación la disposición transitoria 1...
  ```

**Ejemplo real — dropped (Consumidores a19):** el triplete `Téngase…` (→notes) + `Redacción anterior:`
+ texto derogado (→ **dropped**, ni a notes: solapa el vigente y dispararía falsos `note_leak`).

**Ejemplo real — tablas (Haciendas Locales a107, plusvalía):** el BOE escribe tablas de dos formas:
- **Forma A:** `<td><p class="cuerpo_tabla_*">texto</p></td>` → la capta el recorrido de `<p>`.
- **Forma B:** `<td>texto</td>` directo, sin `<p>` → antes se **perdía en silencio**. `is_form_b_table`
  la detecta y `_linearize_table_rows` la convierte **por fila**:
  ```
  [cabeza_tabla]      Periodo de generación | Coeficiente
  [cuerpo_tabla_fila] Inferior a 1 año. | 0,15
  [cuerpo_tabla_fila] 1 año. | 0,15
  ```
  → a `kept`. Respeta el blockquote (tabla forma B vieja dentro de blockquote → `dropped`).

Si en `dropped` cae algo estructural/tabla/artículo → **warning** (posible vigente mal envuelto).

## FASE 7 — `_block_semantics` + `_full_title`: etiquetar el bloque

No se fía del `block_type`; mira si **hay cuerpo de verdad** (`has_retrievable_body` = algún párrafo
cuya clase NO sea un rótulo estructural).

- **`semantic_role`:** precepto→precept · preambulo→preamble · firma→signature · nota_inicial→initial_note ·
  encabezado→(con cuerpo+anexo→annex / con cuerpo→content_heading / sin cuerpo→structural_heading).

**Ejemplo real — cabecera SIN cuerpo (`ci`, CAPÍTULO I de LPAC):**
```
[capitulo_num] CAPÍTULO I
[capitulo_tit] La capacidad de obrar y el concepto de interesado
```
→ todo estructural → `has_retrievable_body=False` → `structural_heading`. `full_title` =
"CAPÍTULO I. La capacidad de obrar y el concepto de interesado".

**Ejemplo real — artículo CON cuerpo (`a3`):**
```
[articulo]  Artículo 3. Capacidad de obrar.   ← rótulo
[parrafo]   A los efectos previstos en esta Ley, tendrán capacidad de obrar...   ← cuerpo
[parrafo_2] a) Las personas físicas o jurídicas...
```
→ `precept`, indexable. `full_title` = "Artículo 3. Capacidad de obrar."

**Ejemplo real — el parser NO se fía del tipo (`s3-3`):** viene como `tipo="precepto"` pero su único
contenido es `[seccion] Sección 3.ª Desistimiento y renuncia` → estructural → sin cuerpo → NO indexable.

Otros flags: `is_annex`, `contains_table`, `is_without_content` (artículo vaciado "(Sin contenido)").

## FASE 8 — `_update_hierarchy`: las "migas de pan"

Estado de 6 niveles (book/title/chapter/section/subsection/annex). Solo los `encabezado` lo actualizan;
**cada nivel reinicia los inferiores**. Cada bloque guarda una copia del estado.

**Ejemplo real — cascada en LPAC:**

| orden | bloque | jerarquía resultante |
|---|---|---|
| 1 | `tpreliminar` (encabezado) | **TÍTULO PRELIMINAR** |
| 2-3 | `a1`, `a2` | TÍTULO PRELIMINAR |
| 4 | `ti` (encabezado) | **TÍTULO I** (sustituye a PRELIMINAR) |
| 5 | `ci` (encabezado) | TÍTULO I › **CAPÍTULO I** |
| 6-11 | `a3`…`a8` | TÍTULO I › CAPÍTULO I |
| 12 | `cii` (encabezado) | TÍTULO I › **CAPÍTULO II** (el título sigue siendo I) |
| 13-15 | `a9`…`a11` | TÍTULO I › CAPÍTULO II |

Sutileza: `s3-3` ("Sección 3.ª") está tipado `precepto`, no `encabezado` → **no** actualiza jerarquía.

## FASE 9 — `build_block_descriptor_fields`: indexable + cita

**`indexable`** = `has_retrievable_body` **Y** `block_type` ∉ {firma, nota_inicial} **Y** hay texto
**Y** no está en cuarentena. Si falla algo → `excluded_reason`.

Casos reales de LPAC:

| bloque | indexable | motivo |
|---|---|---|
| `a3`, `a21`, `preambulo` | True | — |
| `ti`, `ci` (cabeceras) | False | `no_retrievable_body` |
| `s3-3` (sección disfrazada) | False | `no_retrievable_body` |
| `firma` | False | `excluded_type:firma` |
| *(cuarentena)* | False | `temporal_quarantine:<status>` |

**`citation_label`** = `short_title` + ", " + (título del bloque, inicial minúscula):
```
a3        → "Ley 39/2015, artículo 3"
a21       → "Ley 39/2015, artículo 21"
ti        → "Ley 39/2015, TÍTULO I"
preambulo → "Ley 39/2015"   (sin título de bloque)
```
`source_url` = URL del BOE + `#<block_id>`.

## Resumen de las 4 fases

```
Fase 6 reparte el contenido      → kept (cuerpo) / notes / dropped
Fase 7 etiqueta el bloque        → semantic_role, has_body, full_title, flags
Fase 8 sitúa el bloque           → jerarquía (Título › Capítulo › …)
Fase 9 decide visibilidad + cita → indexable + citation_label
```
Cada bloque sale limpio (solo ley vigente), etiquetado, situado y con su cita → listo para `parents`
(el texto) y `document` (el mapa).

---

# Parte C · El bug que esto resolvía: parser antiguo vs nuevo (evidencia real)

## El parser antiguo

Recorría `version.iter("p")` y solo apartaba `nota_pie`/`nota_pie_2` (→ notas). **Todo lo demás iba al
cuerpo**, incluido lo que vivía dentro de `<blockquote>`. Y como solo miraba `<p>`, **las tablas forma
B (`<td>` crudo, sin `<p>`) eran invisibles**. Resultado: **dos fugas opuestas y silenciosas.**

- **Fuga 1 — metía ley vieja/avisos al cuerpo:** todo `<p>` no-nota dentro de un `<blockquote>`
  ("Redacción anterior" + texto derogado, avisos "Téngase…", artículos enteros derogados).
- **Fuga 2 — perdía ley vigente:** las tablas forma B no las veía.

## Evidencia (normas reales)

### Haciendas Locales (`BOE-A-2004-4214`, MVP)

**9 fragmentos colados** (ley vieja/avisos que el antiguo metía como si fueran vigentes):
```
-- bloque a68 --
[parrafo]        Esta última actualización del apartado 4 surte efectos desde el 1 de enero de 2014...   (aviso)
[cita_con_pleca] Redacción hasta 31 de diciembre de 2013:                                                (marcador)
[parrafo]        4. El componente individual de la reducción será, en cada año, la diferencia positiva...  (TEXTO DEROGADO)
-- bloque a72 --
[parrafo]        Téngase en cuenta que se amplía, con vigencia exclusiva para el ejercicio 2023, el plazo...
```
> El marcador no siempre es "Redacción anterior:" — aquí es **"Redacción hasta 31 de diciembre de 2013:"**.
> Otra razón para anclar en el `<blockquote>`, no en el texto del marcador.

**22 filas de tabla recuperadas** por el nuevo (el antiguo las perdía) — `a107`, coeficientes de plusvalía:
```
[cabeza_tabla]      Periodo de generación | Coeficiente
[cuerpo_tabla_fila] Inferior a 1 año. | 0,15
[cuerpo_tabla_fila] 1 año. | 0,15
[cuerpo_tabla_fila] 2 años. | 0,14
```

### Consumidores (`BOE-A-2007-20555`)

**203 fragmentos colados en una sola norma:** artículos enteros derogados (47-52, 114-118…),
redacciones anteriores y avisos. Ejemplo `a19`:
```
[parrafo]        Téngase en cuenta que esta última actualización del apartado 1, establecida por la disposición final 16...
[cita_con_pleca] Redacción anterior:
[parrafo]        "1. Los legítimos intereses económicos y sociales de los consumidores y usuarios..."   (DEROGADO)
```
(0 tablas forma B en esta norma.)

## Escala

| Norma | Fugas (ley vieja/avisos colados) | Tablas recuperadas |
|---|---|---|
| Haciendas Locales (MVP) | 9 | 22 filas (plusvalía `a107`) |
| Consumidores | 203 | 0 |

Corpus-wide (medido en su día): **226 chunks contaminados en 15 de 20 normas** + las tablas forma B
perdidas (~87 celdas vigentes: `a107` de Haciendas Locales y el Anexo II de Tráfico).

## Por qué era grave (las dos fugas a la vez, en silencio)

- La ley vieja es **indistinguible** de la vigente (mismo `parrafo`, tono jurídico, casi idéntica) → el
  RAG podía **citar texto derogado como vigente**, y la respuesta *parece correcta*.
- Las tablas perdidas = **huecos sin respuesta** en datos que el ciudadano consulta (plusvalía, puntos).
- **No saltaba ningún error:** datos malos en silencio. La auditoría `note_leak` era **ciega** (solo
  comparaba contra lo que ya estaba en `modification_notes`).

## Cómo lo cierra el fix

- `classify_version_paragraphs`: `<p>` (o tabla forma B) con ancestro `<blockquote>` → fuera del cuerpo
  (a `notes` si es nota/«Téngase», si no a `dropped`). **Cierra la fuga 1.**
- Captura de tablas forma B (`is_form_b_table` + `_linearize_table_rows`, linealizadas por fila).
  **Cierra la fuga 2.**
- Gates de auditoría (`editorial_leak`, `table_cell_dropped`) + warning "estructura dentro de
  blockquote" → para que un caso nuevo **grite** en vez de corromper el corpus en silencio.
