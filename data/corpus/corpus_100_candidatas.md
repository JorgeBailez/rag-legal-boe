# Corpus 100 — normas candidatas (A1) · VERIFICADO contra boe.es

> **ESTADO:** verificación documental completada (2026-06-22) contra **boe.es** (texto consolidado,
> `https://www.boe.es/buscar/act.php?id=BOE-A-...`). Cada identificador BOE-A, su vigencia y su última
> actualización se confirmaron en la página consolidada oficial. **Pendiente del autor:** decidir
> el recorte de dudosas, los dos "códigos" pesados y el tamaño final. Tras eso → `seed_corpus.json` →
> descarga (`download_boe_raw.py` + `validate_raw_integrity.py`, verificación final).
>
> Marcas: **[T]** tablas · **[A]** anexos · **[R]** reglamento. La detección [T]/[A] es **tentativa**
> (los agentes vieron HTML parcial): se reconfirma contra el XML/PDF en la ingesta.
> Idoneidad: **apta** / **dudosa** (vigente pero poco ciudadana o conflictiva) / **FUERA** (derogada).

## 0 · Núcleo actual del MVP (ya descargado y validado) — 10
| Norma | BOE-A id | Estado |
|---|---|---|
| Ley 39/2015, LPAC | BOE-A-2015-10565 | en producción |
| Ley 40/2015, LRJSP | BOE-A-2015-10566 | en producción |
| Ley 7/1985, Bases Régimen Local | BOE-A-1985-5392 | en producción |
| RDLeg 2/2004, TR Haciendas Locales [T] | BOE-A-2004-4214 | en producción |
| Ley 19/2013, Transparencia | BOE-A-2013-12887 | en producción |
| RD 203/2021, funcionamiento electrónico SP [R] | BOE-A-2021-5032 | en producción |
| Ley 38/2003, General de Subvenciones | BOE-A-2003-20977 | en producción |
| RD 887/2006, Reglamento LGS [R] | BOE-A-2006-13371 | en producción |
| Ley 9/2017, LCSP | BOE-A-2017-12902 | en producción |
| LO 3/2018, LOPDGDD | BOE-A-2018-16673 | en producción |

## 1 · Administrativo / sector público / empleo
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 50/1997, del Gobierno | BOE-A-1997-25336 | VIGENTE | 2025-01-03 | — | apta |
| Ley 47/2003, General Presupuestaria | BOE-A-2003-21614 | VIGENTE | 2022-12-24 | ? | **dudosa** (muy técnica/contable) |
| RDLeg 5/2015, EBEP | BOE-A-2015-11719 | VIGENTE | 2025-07-30 | [T] | apta (empleo público) |
| Ley 29/1998, Contencioso-administrativa | BOE-A-1998-16718 | VIGENTE | 2025-01-03 | — | apta |
| Ley 33/2003, Patrimonio AAPP | BOE-A-2003-20254 | VIGENTE | 2023-05-09 | — | **dudosa** (interés indirecto) |
| Ley 17/2009, libre acceso act. servicios | BOE-A-2009-18731 | VIGENTE (DF 4ª derogada) | 2020-11-12 | — | apta |
| Ley 20/2013, Unidad de Mercado | BOE-A-2013-12888 | VIGENTE (arts. 6/19/20 y DA 10ª inconstitucionales, STC 79/2017) | 2022-09-29 | [A] | **dudosa** (preceptos anulados) |
| Ley 2/2023, protección de informantes | BOE-A-2023-4513 | VIGENTE | 2023-02-21 | — | apta |
| LO 3/1980, del Consejo de Estado | BOE-A-1980-8648 | VIGENTE | 2024-08-02 | — | **dudosa** (bajo uso ciudadano) |

## 2 · Tributario
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 58/2003, General Tributaria | **BOE-A-2003-23186** | VIGENTE | 2024-12-21 | — | apta · id corregido (la pista era de la 47/2003) |
| Ley 35/2006, IRPF | BOE-A-2006-20764 | VIGENTE | 2026-04-29 | [T] | apta |
| Ley 37/1992, IVA | BOE-A-1992-28740 | VIGENTE | 2026-02-28 | [A] | apta |
| Ley 27/2014, Impuesto Sociedades | BOE-A-2014-12328 | VIGENTE | 2026-03-21 | [T] | apta |
| Ley 29/1987, Sucesiones y Donaciones | BOE-A-1987-28141 | VIGENTE | 2022-12-28 | [T] | apta (ojo competencia autonómica) |
| RDLeg 1/1993, ITP y AJD | BOE-A-1993-25359 | VIGENTE | 2026-03-21 | [T] | apta |
| Ley 38/1992, Impuestos Especiales | BOE-A-1992-28741 | VIGENTE | 2026-03-21 | ? | apta (tablas en su rgto RD 1165/1995) |
| RD 1065/2007, Rgto gestión/inspección tributaria | BOE-A-2007-15984 | VIGENTE | 2025-04-02 | [A][R] | apta |
| RD 439/2007, Reglamento del IRPF | BOE-A-2007-6820 | VIGENTE | 2026-02-28 | [T][R] | apta (homonimia con Ley 35/2006) |
| Ley 20/1990, fiscal de cooperativas | BOE-A-1990-30735 | VIGENTE | 2026-04-09 | — | dudosa (nicho) |

## 3 · Laboral / Seguridad Social
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| RDLeg 2/2015, Estatuto de los Trabajadores | BOE-A-2015-11430 | VIGENTE | 2025-12-04 | ? | apta (nuclear) |
| RDLeg 8/2015, LGSS | BOE-A-2015-11724 | VIGENTE | 2026-02-04 | [T]? (bases/tipos cotización) | apta (alta demanda) |
| Ley 31/1995, Prevención Riesgos Laborales | BOE-A-1995-24292 | VIGENTE | 2026-04-09 | ? | apta |
| RDLeg 5/2000, LISOS | BOE-A-2000-15060 | VIGENTE | 2024-05-22 | ? | apta (técnica) |
| Ley 20/2007, Estatuto Trabajo Autónomo | BOE-A-2007-13409 | VIGENTE (algún precepto suspendido) | 2023-03-01 | ? | apta |
| Ley 3/2023, de Empleo | BOE-A-2023-5365 | VIGENTE (deroga RDLeg 3/2015) | 2023-03-01 | [A]? | apta |
| Ley 14/1994, Empresas de Trabajo Temporal | BOE-A-1994-12554 | VIGENTE (parcial) | 2023-03-01 | ? | apta |
| Ley 23/2015, Inspección de Trabajo y SS | BOE-A-2015-8168 | VIGENTE | 2025-01-03 | ? | **dudosa** (orgánica/admva) |
| RD 1620/2011, empleados de hogar [R] | BOE-A-2011-17975 | VIGENTE | 2022-09-08 | ? | apta (alta relevancia) |
| ~~RDLeg 3/2015, Ley de Empleo~~ | ~~BOE-A-2015-11431~~ | **DEROGADO** por Ley 3/2023 | — | — | **FUERA** (duplica la 3/2023) |

## 4 · Vivienda / consumo
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 29/1994, Arrendamientos Urbanos (LAU) | BOE-A-1994-26003 | VIGENTE | 2023-05-25 | — | apta (muy consultada) |
| Ley 49/1960, Propiedad Horizontal | BOE-A-1960-10906 | VIGENTE | 2026-03-21 | — | apta |
| RDLeg 1/2007, TRLGDCU | BOE-A-2007-20555 | VIGENTE | 2026-02-28 | [A] | apta |
| Ley 12/2023, derecho a la vivienda | BOE-A-2023-12203 | VIGENTE | 2026-02-28 | — | apta (reciente) |
| Ley 5/2019, Contratos Crédito Inmobiliario | BOE-A-2019-3814 | VIGENTE | 2023-12-28 | [A] (FEIN/TAE) | apta (hipotecas) |
| Ley 7/1998, Condiciones Generales Contratación | BOE-A-1998-8789 | VIGENTE | 2019-03-16 | — | apta |
| Ley 16/2011, Crédito al Consumo | BOE-A-2011-10970 | VIGENTE | 2014-03-28 | [T][A] | apta |
| Ley 22/2007, comercializ. a distancia serv. financieros | BOE-A-2007-13411 | VIGENTE (solo art. 12 derogado) | 2018-11-24 | — | apta (no citar art. 12) |
| Ley 4/2012, aprovechamiento por turno (multipropiedad) | BOE-A-2012-9111 | VIGENTE | 2025-01-03 | [A] | apta |
| Ley 28/1998, venta a plazos bienes muebles | BOE-A-1998-16717 | VIGENTE | 2011-10-11 | — | apta |

## 5 · Civil / familia / justicia
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 1/2000, Enjuiciamiento Civil (LEC) | BOE-A-2000-323 | VIGENTE | 2025-02-28 | ? | apta (extensa, >800 arts.) |
| Ley 15/2015, Jurisdicción Voluntaria | BOE-A-2015-7391 | VIGENTE | 2025-01-03 | — | apta |
| Ley 8/2021, apoyo a personas con discapacidad | BOE-A-2021-9233 | VIGENTE | 2024-11-14 | — | **dudosa** (poco autónoma: son mandatos de reforma de CC/LEC) |
| Ley 20/2011, Registro Civil | BOE-A-2011-12628 | VIGENTE | 2025-01-03 | — | apta (muy consultada) |
| Ley 41/2003, protección patrimonial discapacidad | BOE-A-2003-21053 | VIGENTE | 2023-05-25 | — | apta (breve) |
| Ley 5/2012, mediación civil y mercantil | BOE-A-2012-9112 | VIGENTE | 2025-01-03 | — | apta |
| Ley 29/2015, cooperación jurídica internacional | BOE-A-2015-8564 | VIGENTE | 2022-09-06 | — | dudosa (nicho) |
| **Código Civil** (RD 24-jul-1889) | BOE-A-1889-4763 | VIGENTE | 2025-01-03 | — | **diferir/piloto** (~1.976 arts., formato 1889 → riesgo parser) |

## 6 · Sanidad
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 14/1986, General de Sanidad | BOE-A-1986-10499 | VIGENTE (parcialmente derogada) | 2023-03-23 | [A]? | apta (marcar preceptos derogados) |
| Ley 41/2002, autonomía del paciente | BOE-A-2002-22188 | VIGENTE | 2023-03-01 | — | apta (muy relevante) |
| Ley 16/2003, cohesión y calidad del SNS | BOE-A-2003-10715 | VIGENTE | 2024-10-31 | [A]? | apta |
| Ley 33/2011, General de Salud Pública | BOE-A-2011-15623 | VIGENTE | 2025-07-29 | [A]? | apta |
| RDLeg 1/2015, garantías y uso racional de medicamentos | BOE-A-2015-8343 | VIGENTE | 2026-05-27 | [A]? | **dudosa** (muy técnica) |
| Ley 44/2003, ordenación profesiones sanitarias | BOE-A-2003-21340 | VIGENTE | 2021-06-05 | — | apta |

## 7 · Educación / ciencia
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| LO 2/2006, de Educación (LOE+LOMLOE) | BOE-A-2006-7899 | VIGENTE | 2024-06-08 | [A]? | apta (nuclear) |
| LO 2/2023, del Sistema Universitario (LOSU) | BOE-A-2023-7500 | VIGENTE | 2024-08-02 | ? | apta |
| Ley 14/2011, de la Ciencia | BOE-A-2011-9617 | VIGENTE | 2023-01-11 | ? | apta |
| LO 3/2022, integración de la FP | BOE-A-2022-5139 | VIGENTE | 2024-06-08 | ? | apta |

## 8 · Extranjería / asilo
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| LO 4/2000, extranjería | BOE-A-2000-544 | VIGENTE | 2025-03-19 | — | apta (muy consultada) |
| Ley 12/2009, asilo y protección subsidiaria | BOE-A-2009-17242 | VIGENTE | 2023-03-01 | — | apta |
| **RD 1155/2024, nuevo Reglamento de extranjería** [R][A] | **BOE-A-2024-24099** | VIGENTE (desde 2025-05-20) | 2026-04-15 | [A]? | apta · **sustituye** al RD 557/2011 |
| ~~RD 557/2011, Reglamento extranjería~~ | ~~BOE-A-2011-7703~~ | **DEROGADO** por RD 1155/2024 | — | — | **FUERA** |
| Ley 19/2015, reforma admva. Justicia/Registro Civil | BOE-A-2015-7851 | VIGENTE | 2015-07-14 | — | **dudosa** (instrumental/modificativa) |

## 9 · Tráfico / movilidad
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| RDLeg 6/2015, Ley de Tráfico y Seguridad Vial | BOE-A-2015-11722 | VIGENTE | 2025-12-04 | [A] | apta |
| RDLeg 8/2004, RC y seguro circulación | BOE-A-2004-18911 | VIGENTE (mod. Ley 5/2025) | ~2025-07 | [A] (baremo) | apta (reclamaciones) |
| ~~Ley 35/2015, valoración de daños (baremo)~~ | ~~BOE-A-2015-10197~~ | VIGENTE como disposición, **SIN texto consolidado propio** | — | — | **FUERA** (404 en la API consolidada el 2026-06-22: es ley de reforma; el baremo vive en el anexo del RDLeg 8/2004, ya en el corpus) |
| RD 1428/2003, Reglamento General de Circulación [R] | BOE-A-2003-23514 | VIGENTE (mod. RD 465/2025) | ~2025-06 | [T][A]? | apta (señales) |

## 10 · Digital / telecomunicaciones
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 34/2002, LSSI-CE | BOE-A-2002-13758 | VIGENTE | 2025-01-23 | [A] | apta |
| Ley 11/2022, General de Telecomunicaciones | BOE-A-2022-10757 | VIGENTE | 2025-12-27 | [A] | apta |
| Ley 56/2007, impulso sociedad de la información | BOE-A-2007-22440 | VIGENTE (muy fragmentaria) | 2024-12-21 | — | **dudosa** (norma fragmentaria) |
| Ley 6/2020, servicios electrónicos de confianza | BOE-A-2020-14046 | VIGENTE | 2023-05-09 | — | apta |
| Ley 13/2022, General de Comunicación Audiovisual | BOE-A-2022-11311 | VIGENTE | 2022-07-08 | — | apta |
| ~~Ley 9/2014, General de Telecomunicaciones~~ | ~~BOE-A-2014-4950~~ | **DEROGADA** por Ley 11/2022 | — | — | **FUERA** (duplica la 11/2022) |

## 11 · Medio ambiente / energía
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| Ley 21/2013, Evaluación Ambiental | BOE-A-2013-12913 | VIGENTE | 2025-11-06 | [A] (anexos I–VI) | apta |
| Ley 7/2021, cambio climático y transición energética | BOE-A-2021-8447 | VIGENTE | 2025-12-04 | — | apta (articulado limpio) |
| Ley 24/2013, del Sector Eléctrico | BOE-A-2013-13645 | VIGENTE | 2026-03-21 | [A] | apta (extensa) |
| Ley 7/2022, residuos y suelos contaminados | BOE-A-2022-5809 | VIGENTE | 2025-04-02 | [T][A] (16 anexos) | **dudosa** (anexos tabulares → riesgo chunking) |
| Ley 42/2007, Patrimonio Natural y Biodiversidad | BOE-A-2007-21490 | VIGENTE | 2020-12-31 | [A] (8 anexos) | apta |
| RDLeg 1/2001, TR Ley de Aguas | BOE-A-2001-14276 | VIGENTE | 2023-12-28 | — | apta (articulado limpio) |

## 12 · Asociaciones / fundaciones / seguridad / igualdad / dependencia
| Norma | BOE-A id | Vigencia | Últ. act. | [T]/[A] | Idoneidad |
|---|---|---|---|---|---|
| LO 1/2002, Derecho de Asociación | BOE-A-2002-5852 | VIGENTE | 2025-06-28 | — | apta |
| Ley 50/2002, de Fundaciones | BOE-A-2002-25180 | VIGENTE | 2024-08-02 | — | apta |
| Ley 49/2002, mecenazgo | BOE-A-2002-25039 | VIGENTE | 2023-12-20 | [A] | apta |
| LO 4/2015, seguridad ciudadana | BOE-A-2015-3442 | VIGENTE (matices STC 172/2020, 13/2021) | 2021-02-23 | — | apta (reflejar matices TC) |
| Ley 15/2022, igualdad de trato y no discriminación | BOE-A-2022-11589 | VIGENTE | 2022-07-13 | — | apta |
| LO 3/2007, igualdad efectiva mujeres y hombres | BOE-A-2007-6115 | VIGENTE | 2024-08-02 | — | apta |
| Ley 39/2006, dependencia | BOE-A-2006-21990 | VIGENTE | 2025-10-22 | — | apta (altísimo interés) |

## 13 · Grandes ausentes — DECISIÓN del autor (2026-06-22)
| Norma | BOE-A id | Vigencia | Tamaño | Decisión |
|---|---|---|---|---|
| Constitución Española 1978 | BOE-A-1978-31229 | VIGENTE | ~169 arts. | **AÑADIR** (riesgo nulo, valor altísimo) |
| LO 10/1995, Código Penal | BOE-A-1995-25444 | VIGENTE | ~639 arts. | **AÑADIR** (formato moderno, alto interés; cubierto por el disclaimer estático) |
| Ley Hipotecaria (1946) | BOE-A-1946-2453 | VIGENTE | ~329 arts. | No en esta fase (solo era para forzar el 100 exacto) |
| RDLeg 1/2020, TR Ley Concursal (TRLC) | BOE-A-2020-4859 | VIGENTE | ~755 arts. | No en esta fase (pesada y técnica-mercantil) |
| LO 6/1985, Poder Judicial (LOPJ) | BOE-A-1985-12666 | VIGENTE | ~800+ arts. | **DIFERIR** (enorme, procesal, reforma constante) |
| Código Civil (1889) | BOE-A-1889-4763 | VIGENTE | ~1.976 arts. | **DIFERIR / piloto** (formato 1889 = máximo riesgo de parser; fuera del camino crítico) |

---

## Resumen de la verificación
- **Identificadores:** 0 sin confirmar. Todos verificados en el consolidado de boe.es. **1 corregido:**
  Ley 58/2003 LGT = `BOE-A-2003-23186` (la pista original era la de la 47/2003).
- **Derogadas detectadas → FUERA (3):** RDLeg 3/2015 (→ Ley 3/2023), Ley 9/2014 (→ Ley 11/2022) y
  **RD 557/2011 (→ RD 1155/2024, hallazgo nuevo: hay reglamento de extranjería de 2024)**.
- **Caída en la descarga → FUERA (1):** Ley 35/2015 (`BOE-A-2015-10197`) dio **404 en la API de
  legislación consolidada** (2026-06-22): es ley de reforma sin consolidado propio; su baremo vive en
  el anexo del RDLeg 8/2004 (ya en el corpus). El id es correcto, pero no cumple "tener consolidado".
- **Vigentes-aptas (núcleo + secciones 1-12, sin las FUERA):** 10 núcleo + 82 nuevas = **92**.
- **Dudosas (vigentes, decisión de recorte):** Ley 47/2003, Ley 33/2003, LO 3/1980, Ley 20/1990,
  Ley 23/2015, Ley 8/2021, Ley 29/2015, RDLeg 1/2015, Ley 19/2015, Ley 56/2007, Ley 7/2022,
  Ley 20/2013 (preceptos anulados). ~12.
- **Cuotas para el flagship:** [T]/[A] abundan en tributario y tráfico (estrés de parser); homonimia
  de artículos garantizada por el volumen; el RD 439/2007 (rgto) ↔ Ley 35/2006 da cruces reglamento-ley.

## Decisión final (2026-06-22) — corpus 92, bajo riesgo
- **Dentro:** núcleo 10 + todas las VIGENTES de §1-12 (las "dudosas" entran pero se marcan para no
  redactar gold sobre preceptos anulados) + **Constitución** + **Código Penal**.
- **Fuera (derogadas):** RDLeg 3/2015, Ley 9/2014 y RD 557/2011 → este último **sustituido por
  RD 1155/2024**. **+ Ley 35/2015** (sin consolidado propio, cayó en la descarga; baremo en RDLeg 8/2004).
- **Diferido:** Código Civil y LOPJ (riesgo de parser / peso); Ley Hipotecaria y TRLC no entran ahora.
- **Total descargado y procesado: 92 normas** (seed pedía 93; la Ley 35/2015 cayó por 404 en la API
  consolidada → corpus efectivo 92). Cumple el "~100" del PLAN sin meter el riesgo del CC en el cierre.

## Siguiente paso (A2)
1. Lista confirmada (decisión 2026-06-22).
2. Seed nuevo `data/corpus/seed_corpus_ampliado.json` (92 normas; `seed_corpus.json` MVP-10 intacto).
   `expected_rank` de la Constitución = `"Constitución"` (es documental: `load_seed_corpus` no lo
   valida; el rank real se lee de los metadatos al descargar). Scripts del flujo con flag `--seed`
   (default = MVP-10) y `build_corpus` escribe `verification_report_ampliado.json` (no pisa el del 10).
3. **Descarga (con red, en una máquina de trabajo)** — baja raw + manifest, verifica y procesa las 93:
   ```bash
   uv run --locked python scripts/build_corpus.py --seed data/corpus/seed_corpus_ampliado.json
   uv run --locked python scripts/validate_raw_integrity.py --seed data/corpus/seed_corpus_ampliado.json
   uv run python scripts/validate_mvp_corpus.py --seed data/corpus/seed_corpus_ampliado.json --strict
   uv run python scripts/audit_corpus.py --strict   # recorre lo procesado; --seed no aplica
   ```
   Las que no cumplan criterios se reportan (no se sustituyen). Vigilar el parseo de las [T]/[A]
   densas (tributario, tráfico, Ley 7/2022 con 16 anexos) y los errores de proceso.
