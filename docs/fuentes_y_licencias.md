# Fuentes y licencias

Endpoints del BOE, condiciones de uso y referencias. Documento vivo.

## Estado: ingesta raw confirmada

Endpoints confirmados e implementados en `src/boe/client.py` para la descarga raw.

## API BOE — legislación consolidada

- FAQ y documentación: `boe.es/datosabiertos/faq/consolidada.php`
- Base configurable vía `BOE_API_BASE` (default `https://www.boe.es/datosabiertos/api`).
- Endpoints usados por la ingesta raw (relativos a la base, con `{id}` = identificador BOE):
  - `/legislacion-consolidada/id/{id}` — documento completo (`full`)
  - `/legislacion-consolidada/id/{id}/metadatos` — metadatos (`metadatos`)
  - `/legislacion-consolidada/id/{id}/analisis` — análisis (`analisis`)
  - `/legislacion-consolidada/id/{id}/metadata-eli` — metadatos ELI (`metadata_eli`)
  - `/legislacion-consolidada/id/{id}/texto` — texto consolidado (`texto`)
  - `/legislacion-consolidada/id/{id}/texto/indice` — índice de bloques (`indice`)
- Endpoint de referencia para fases posteriores (no usado aún):
  - `/legislacion-consolidada/id/{id}/texto/bloque/{id_bloque}` — bloque concreto.
- No se usan endpoints antiguos tipo `/legislacion/documento/{id}` ni scraping HTML.

## Identificadores

- Clave de acceso: `BOE-A-YYYY-NNNNN` (ej. norma MVP: `BOE-A-2015-10565`).
- ELI: se conserva como metadato/URL permanente, no como clave de acceso.

## Licencias y condiciones de uso

El proyecto reutiliza la legislación consolidada que el BOE publica como datos abiertos. Esa
reutilización se rige por la Ley 37/2007, sobre reutilización de la información del sector público, y
por el Real Decreto 1495/2011, que la desarrolla para el sector público estatal, en los términos del
aviso legal del BOE. Las condiciones generales permiten la reutilización con fines comerciales y no
comerciales siempre que se cumplan dos obligaciones básicas:

- no desnaturalizar el sentido de la información, y
- citar como fuente al BOE e indicar la fecha de la última actualización del documento reutilizado.

Cuando la información contenga datos personales, su reutilización debe respetar el Reglamento General
de Protección de Datos y la Ley Orgánica 3/2018.

Referencias oficiales:

- Aviso legal del BOE: <https://www.boe.es/informacion/aviso_legal/index.php>
- Real Decreto 1495/2011, de reutilización en el sector público estatal:
  <https://www.boe.es/buscar/act.php?id=BOE-A-2011-17560>
- Datos abiertos del BOE, preguntas frecuentes sobre legislación consolidada:
  <https://www.boe.es/datosabiertos/faq/consolidada.php>

## Aviso

Los textos consolidados del BOE tienen carácter informativo y no valor jurídico oficial. Para
cualquier uso con efectos jurídicos hay que remitirse siempre a la publicación oficial.
