# Análisis del parser y del corpus

Esta carpeta documenta, con detalle y ejemplos reales, cómo se interpreta el texto legal del BOE antes de indexarlo: qué se descarga, cómo lo desmonta el parser y qué decisiones se tomaron sobre qué conservar. Sirve como material de onboarding para entender el preprocesado y como registro de decisiones que, de otro modo, quedarían implícitas en el código.

Los análisis se elaboraron durante la primera fase del proyecto, sobre una muestra de 20 normas. Los recuentos concretos (cuántos bloques de cada tipo, cuántas tablas, etc.) corresponden a esa muestra; la arquitectura del parser que describen es la misma que se usa sobre el corpus actual de 92 normas. Para el estado y los resultados del proyecto, ver `decisiones_de_diseno.md`.

## Documentos

- [01_parser_flujo_completo.md](01_parser_flujo_completo.md) — El parser explicado desde cero: qué es un fichero XML, qué se descarga del BOE, las once fases del flujo y los tres artículos de salida, con una profundización en las fases centrales (clasificación de párrafos, resolución temporal, jerarquía, indexabilidad) sobre normas reales.

- [02_blockquotes.md](02_blockquotes.md) — La anatomía del aparato editorial del BOE (las citas y notas que rodean al texto vigente): sus familias, los casos límite y la decisión de qué conservar para la búsqueda y qué descartar.

- [03_tipos_de_bloque.md](03_tipos_de_bloque.md) — Los cinco tipos de bloque del BOE (precepto, encabezado, preámbulo, firma y nota inicial), su peso para la recuperación y una evaluación de qué mejoras de parser merecían la pena y cuáles no.

- [04_tablas.md](04_tablas.md) — El tratamiento de las tablas dentro de las normas: dónde aparecen, las dos formas en que el BOE las codifica y cómo se decide indexarlas o no.

## Para qué sirve

- Onboarding de quien se incorpore al proyecto.
- Registro de las decisiones de preprocesado que hoy viven en el código.
- Base para la sección de parser y preprocesado de la memoria.
