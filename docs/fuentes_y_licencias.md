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

Se reutiliza la información de datos abiertos del BOE (legislación consolidada). La
legislación consolidada reutilizada tiene **carácter informativo**: no sustituye al texto
oficial publicado en el BOE. Pendiente: citar formalmente las condiciones de reutilización
de datos abiertos del BOE.

## Aviso

Los textos consolidados del BOE tienen carácter informativo y no valor jurídico oficial;
remitir siempre a la publicación oficial.
