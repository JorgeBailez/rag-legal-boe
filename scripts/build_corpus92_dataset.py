"""Genera el dataset de evaluación corpus92_v1 (questions.jsonl + judgments.jsonl).

Consolida las preguntas redactadas (lote de calibración del asistente + 6 anotadores en paralelo),
GROUNDED contra el corpus real (block_id/full_title verbatim). Cada judgment lleva parent_id +
relevance (diana known-item) + evidencia anotada (`paragraph_orders` + `quote` literal byte-a-byte,
fundida desde `_evidence/*.json`). Verificación del asistente (política 2026-06-13): tras pasar la
verificación programática (parent vigente, orders existentes, quote literal, multi_parent>=2) +
revisión semántica par a par, se promueven a `reviewed` las 81 in-corpus y las 24 OOC limpiamente
ajenas al corpus (6 OOC con solape parcial quedan `draft`, ver OOC_DRAFT). `reviewed` certifica la
CORRECCIÓN de la diana (precisión); la COMPLETITUD (no faltan parents relevantes) la dará el pooling
de 3 retrievers, todavía pendiente. Splits provisionales (todo `development`); el split dev/test
definitivo y la fusión con el banco MVP (dense_retrieval_v1) son un paso aparte.

Reproducible: `uv run python scripts/build_corpus92_dataset.py`. Valida al final contra el corpus.
"""
# ruff: noqa: E501  -- tabla de datos: las queries son lenguaje natural en una línea por legibilidad.

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/evaluation/corpus92_v1")

# norm_id por norma (para construir parent_id = f"{norm}__{block_id}")
LAU = "BOE-A-1994-26003"
ET = "BOE-A-2015-11430"
TRF = "BOE-A-2015-11722"
CIN = "BOE-A-2019-3814"
DEP = "BOE-A-2006-21990"
LGT = "BOE-A-2003-23186"
IVA = "BOE-A-1992-28740"
IS = "BOE-A-2014-12328"
ISD = "BOE-A-1987-28141"
RIRPF = "BOE-A-2007-6820"
ITP = "BOE-A-1993-25359"
COOP = "BOE-A-1990-30735"
PAC = "BOE-A-2002-22188"
LEC = "BOE-A-2000-323"
MED = "BOE-A-2015-8343"
CJI = "BOE-A-2015-8564"
SPUB = "BOE-A-2011-15623"
MEDIA = "BOE-A-2012-9112"
JV = "BOE-A-2015-7391"
RC = "BOE-A-2011-12628"
RDEXT = "BOE-A-2024-24099"
LOEX = "BOE-A-2000-544"
ASILO = "BOE-A-2009-17242"
LOSU = "BOE-A-2023-7500"
LOE = "BOE-A-2006-7899"
FP = "BOE-A-2022-5139"
CIEN = "BOE-A-2011-9617"
LSSI = "BOE-A-2002-13758"
CONF = "BOE-A-2020-14046"
CE = "BOE-A-1978-31229"
CP = "BOE-A-1995-25444"
LGSS = "BOE-A-2015-11724"
PRL = "BOE-A-1995-24292"
AUT = "BOE-A-2007-13409"
TRLG = "BOE-A-2007-20555"
VP = "BOE-A-1998-16717"
CC_CONS = "BOE-A-2011-10970"
AGUAS = "BOE-A-2001-14276"
CLIMA = "BOE-A-2021-8447"
ELEC = "BOE-A-2013-13645"
FUND = "BOE-A-2002-25180"
IGMH = "BOE-A-2007-6115"


def p(norm: str, block: str) -> str:
    return f"{norm}__{block}"


# (issue_family_id, query, query_style, answer_scope, difficulty, [(parent_id, relevance), ...])
INCORPUS = [
    # --- lote de calibración del asistente ---
    (
        "lau_duracion",
        "Firmé un alquiler de vivienda por un año y el casero dice que al acabar me tengo que ir. ¿Tengo derecho a quedarme más tiempo?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(LAU, "a9"), 2), (p(LAU, "a10"), 1)],
    ),
    (
        "lau_duracion",
        "¿Cuál es el plazo mínimo de duración y la prórroga obligatoria del arrendamiento de vivienda habitual?",
        "conceptual",
        "multi_parent",
        "media",
        [(p(LAU, "a9"), 2), (p(LAU, "a10"), 1)],
    ),
    (
        "lau_duracion",
        "¿Qué establece el artículo 9 de la Ley de Arrendamientos Urbanos?",
        "directa_articulo",
        "single_parent",
        "facil",
        [(p(LAU, "a9"), 2)],
    ),
    (
        "et_vacaciones",
        "¿Cuántos días de vacaciones pagadas me corresponden al año?",
        "ciudadana",
        "single_parent",
        "facil",
        [(p(ET, "a38"), 2)],
    ),
    (
        "et_vacaciones",
        "¿Qué dice el artículo 38 del Estatuto de los Trabajadores?",
        "directa_articulo",
        "single_parent",
        "facil",
        [(p(ET, "a38"), 2)],
    ),
    (
        "trafico_art3",
        "¿Qué regula el artículo 3 del texto refundido de la Ley de Tráfico (RDLeg 6/2015)?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(TRF, "a3"), 2)],
    ),
    (
        "credinmob_tae",
        "¿Cómo se calcula la TAE de un préstamo hipotecario?",
        "lexica",
        "multi_parent",
        "media",
        [(p(CIN, "ar-8"), 2)],
    ),
    (
        "trafico_alcohol",
        "Me pararon en un control y había bebido; ¿qué tasa de alcohol permite la ley al conducir?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(TRF, "a14"), 1)],
    ),
    (
        "dependencia_concepto",
        "¿Qué se considera una situación de dependencia y qué grados existen?",
        "conceptual",
        "multi_parent",
        "media",
        [(p(DEP, "a2"), 1), (p(DEP, "a26"), 2)],
    ),
    (
        "dependencia_reconocimiento",
        "¿Qué pasos sigo para que me reconozcan oficialmente la dependencia?",
        "procedimental",
        "single_parent",
        "media",
        [(p(DEP, "a28"), 2)],
    ),
    (
        "et_prueba_vs_temporal",
        "¿En qué se diferencia el periodo de prueba de la duración de un contrato temporal según el Estatuto de los Trabajadores?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(ET, "a14"), 2), (p(ET, "a15"), 2)],
    ),
    (
        "lau_desistimiento",
        "Quiero dejar el piso de alquiler antes de que acabe el contrato; ¿puedo, y tiene penalización?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LAU, "a11"), 2)],
    ),
    (
        "lau_renta",
        "¿Cuánto puede subirme el casero la renta del alquiler cada año?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LAU, "a18"), 2)],
    ),
    (
        "credinmob_demora",
        "¿Qué regula el artículo 25 de la Ley 5/2019 sobre intereses de demora en hipotecas?",
        "directa_articulo",
        "single_parent",
        "facil",
        [(p(CIN, "ar-25"), 2)],
    ),
    # --- tributario ---
    (
        "lgt_recargo_extemp",
        "Si presento una autoliquidación fuera de plazo pero antes de que Hacienda me lo reclame, ¿qué recargo me toca pagar?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LGT, "a27"), 2)],
    ),
    (
        "lgt_recargo_extemp",
        "¿Cuál es el porcentaje del recargo por declaración extemporánea sin requerimiento previo regulado en la LGT?",
        "lexica",
        "single_parent",
        "media",
        [(p(LGT, "a27"), 2)],
    ),
    (
        "lgt_recargo_extemp",
        "¿Qué establece el artículo 27 de la Ley General Tributaria?",
        "directa_articulo",
        "single_parent",
        "facil",
        [(p(LGT, "a27"), 2)],
    ),
    (
        "lgt_prescripcion",
        "¿En cuántos años prescribe el derecho de Hacienda a liquidar una deuda tributaria?",
        "ciudadana",
        "single_parent",
        "facil",
        [(p(LGT, "a66"), 2)],
    ),
    (
        "lgt_prescripcion",
        "¿Cuál es el plazo de prescripción del derecho a determinar la deuda tributaria conforme a la LGT?",
        "conceptual",
        "single_parent",
        "media",
        [(p(LGT, "a66"), 2)],
    ),
    (
        "iva_tipo_general",
        "¿A qué tipo general se aplica el IVA en España?",
        "lexica",
        "single_parent",
        "facil",
        [(p(IVA, "a90"), 2)],
    ),
    (
        "iva_tipo_general",
        "¿Qué dice el artículo 90 de la Ley del IVA?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(IVA, "a90"), 2)],
    ),
    (
        "is_tipo_gravamen",
        "¿Cuál es el tipo general de gravamen en el Impuesto sobre Sociedades?",
        "lexica",
        "single_parent",
        "facil",
        [(p(IS, "a29"), 2)],
    ),
    (
        "isd_ajuar",
        "¿Cómo se valora el ajuar doméstico a efectos del Impuesto sobre Sucesiones y Donaciones?",
        "conceptual",
        "single_parent",
        "media",
        [(p(ISD, "a15"), 2)],
    ),
    (
        "irpf_retencion_local",
        "¿Qué retención debo practicar al pagar el alquiler de un local de negocio (IRPF)?",
        "procedimental",
        "single_parent",
        "media",
        [(p(RIRPF, "a100"), 2)],
    ),
    (
        "itpajd_ajd_escritura",
        "¿Cómo se grava en Actos Jurídicos Documentados (AJD) la primera copia de una escritura notarial?",
        "lexica",
        "single_parent",
        "dificil",
        [(p(ITP, "a31"), 2)],
    ),
    (
        "coop_proteccion_bin",
        "¿Qué condiciones debe cumplir una cooperativa para ser fiscalmente protegida y en qué consiste el límite del 70% al compensar bases imponibles negativas en el IS?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(COOP, "art6"), 2), (p(IS, "a26"), 1)],
    ),
    # --- civil / justicia + sanidad ---
    (
        "paciente_consentimiento",
        "Si me van a operar, ¿el médico tiene que pedirme permiso por escrito o basta con que me lo explique de palabra?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(PAC, "a8"), 2)],
    ),
    (
        "paciente_consentimiento",
        "¿En qué supuestos exige la ley que el consentimiento del paciente conste por escrito y no solo verbalmente?",
        "procedimental",
        "single_parent",
        "media",
        [(p(PAC, "a8"), 2)],
    ),
    (
        "paciente_consentimiento",
        "¿Qué dice el artículo 8 de la Ley 41/2002 de autonomía del paciente?",
        "directa_articulo",
        "single_parent",
        "facil",
        [(p(PAC, "a8"), 2)],
    ),
    (
        "lec_costas",
        "Si pierdo un juicio civil, ¿me obligan siempre a pagar las costas del que ganó?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LEC, "a394"), 2)],
    ),
    (
        "lec_costas",
        "¿Conforme a qué criterio se imponen las costas a una de las partes en la primera instancia de un proceso declarativo?",
        "conceptual",
        "single_parent",
        "media",
        [(p(LEC, "a394"), 2)],
    ),
    (
        "medicamentos_sustitucion",
        "¿Qué establece el artículo 89 de la ley de garantías y uso racional de los medicamentos?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(MED, "a89"), 2)],
    ),
    (
        "medicamentos_sustitucion",
        "¿Puede un farmacéutico sustituir el medicamento de mi receta por otro si no hay existencias?",
        "lexica",
        "single_parent",
        "media",
        [(p(MED, "a89"), 2)],
    ),
    (
        "coopjur_exequatur",
        "¿Qué regula el artículo 42 de la Ley 29/2015 de cooperación jurídica internacional en materia civil?",
        "directa_articulo",
        "single_parent",
        "dificil",
        [(p(CJI, "a42"), 2)],
    ),
    (
        "paciente_instrucciones_previas",
        "¿Qué son las instrucciones previas o voluntades anticipadas y quién puede otorgarlas?",
        "lexica",
        "single_parent",
        "media",
        [(p(PAC, "a11"), 2)],
    ),
    (
        "intimidad_salud_dos_leyes",
        "¿Qué leyes me amparan el derecho a la intimidad y confidencialidad de mis datos de salud, tanto como paciente como frente a actuaciones de salud pública?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(PAC, "a7"), 2), (p(SPUB, "a7"), 2)],
    ),
    (
        "mediacion_titulo_ejecutivo",
        "¿Cómo se convierte en ejecutable un acuerdo de mediación y qué relación tiene con la conciliación previa al pleito?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(MEDIA, "a25"), 2), (p(JV, "a139"), 1)],
    ),
    (
        "registrocivil_nombre",
        "¿Puedo elegir libremente el nombre de mi hijo al inscribirlo o hay límites en el Registro Civil?",
        "ciudadana",
        "multi_parent",
        "facil",
        [(p(RC, "a51"), 2), (p(RC, "a52"), 1)],
    ),
    # --- extranjería + educación + digital ---
    (
        "extranjeria_arraigo",
        "¿Cuántos tipos de arraigo existen para regularizarse en España y cuánto dura cada autorización?",
        "conceptual",
        "multi_parent",
        "media",
        [(p(RDEXT, "a1-37"), 2), (p(RDEXT, "a1-38"), 1)],
    ),
    (
        "extranjeria_arraigo",
        "Llevo dos años en España sin papeles, ¿qué modalidades de arraigo podría pedir y qué me exigen?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(RDEXT, "a1-37"), 2), (p(RDEXT, "a1-38"), 2)],
    ),
    (
        "extranjeria_nie",
        "¿Qué es el NIE y quién lo asigna?",
        "lexica",
        "single_parent",
        "facil",
        [(p(RDEXT, "a2-17"), 2)],
    ),
    (
        "extranjeria_cie",
        "¿Cuánto tiempo como máximo puede estar internada una persona en un centro de internamiento de extranjeros?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LOEX, "a62"), 2)],
    ),
    (
        "asilo_subsidiaria",
        "¿Qué es la protección subsidiaria y en qué se diferencia de la condición de refugiado?",
        "comparativa",
        "multi_parent",
        "media",
        [(p(ASILO, "a4"), 2), (p(ASILO, "a3"), 2), (p(ASILO, "a10"), 1)],
    ),
    (
        "asilo_frontera_plazo",
        "Si pido asilo en un puesto fronterizo, ¿en cuántos días tienen que resolver mi solicitud?",
        "procedimental",
        "single_parent",
        "dificil",
        [(p(ASILO, "a21"), 2)],
    ),
    (
        "becas_losu_vs_loe",
        "¿Cómo regula la ley universitaria las becas y ayudas al estudio frente a cómo lo hace la ley de educación?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(LOSU, "a3-4"), 2), (p(LOE, "a83"), 2)],
    ),
    (
        "fp_dual",
        "¿Qué significa que la Formación Profesional tenga carácter dual y a qué grados se aplica?",
        "lexica",
        "multi_parent",
        "media",
        [(p(FP, "a5-7"), 2), (p(FP, "a2-10"), 1)],
    ),
    (
        "ciencia_predoctoral",
        "¿Quién puede firmar un contrato predoctoral y qué requisitos académicos exige la Ley de la Ciencia?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(CIEN, "a21"), 2)],
    ),
    (
        "lssi_comunicaciones",
        "¿Puede una empresa enviarme publicidad por correo electrónico sin que yo lo haya autorizado antes?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(LSSI, "a21"), 2), (p(LSSI, "a22"), 1)],
    ),
    (
        "lssi_info_general",
        "Según el artículo 10 de la LSSI, ¿qué información general está obligado a mostrar el prestador de servicios en su web?",
        "directa_articulo",
        "single_parent",
        "dificil",
        [(p(LSSI, "a10"), 2)],
    ),
    (
        "confianza_certificado",
        "Según la ley de servicios electrónicos de confianza, ¿cuánto puede durar como máximo la vigencia de un certificado electrónico cualificado?",
        "directa_articulo",
        "single_parent",
        "dificil",
        [(p(CONF, "a4"), 2)],
    ),
    # --- Constitución + Código Penal + admin/laboral ---
    (
        "ce_detencion",
        "Si la policía me detiene, ¿durante cuánto tiempo pueden tenerme retenido y qué me tienen que decir?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(CE, "a17"), 2)],
    ),
    (
        "ce_intimidad_domicilio",
        "¿Qué dice la Constitución sobre la protección del honor, la intimidad y la inviolabilidad del domicilio frente a registros?",
        "conceptual",
        "single_parent",
        "media",
        [(p(CE, "a18"), 2)],
    ),
    (
        "ce_art20",
        "¿Qué reconoce el artículo 20 de la Constitución española?",
        "directa_articulo",
        "single_parent",
        "dificil",
        [(p(CE, "a20"), 2)],
    ),
    (
        "ce_vivienda_educacion",
        "¿Reconoce la Constitución como derecho el acceso a una vivienda digna y, por otro lado, a la educación? ¿En qué artículos?",
        "comparativa",
        "multi_parent",
        "media",
        [(p(CE, "a47"), 2), (p(CE, "a27"), 2)],
    ),
    (
        "cp_hurto",
        "Me han quitado la cartera sin que me diera cuenta, sin violencia. ¿Eso qué delito es y de qué depende la pena?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(CP, "a234"), 2)],
    ),
    (
        "cp_estafa",
        "Me engañaron para que transfiriera dinero a cambio de algo que nunca recibí. ¿Qué delito describe esa conducta?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(CP, "a248"), 2)],
    ),
    (
        "cp_omision_socorro",
        "¿Es delito no ayudar a una persona que está en peligro grave en la calle si yo podía ayudarla sin riesgo?",
        "conceptual",
        "single_parent",
        "dificil",
        [(p(CP, "a195"), 2)],
    ),
    (
        "cp_descubrimiento_secretos",
        "¿Qué conducta tipifica el artículo 197 del Código Penal?",
        "directa_articulo",
        "single_parent",
        "dificil",
        [(p(CP, "a197"), 2)],
    ),
    (
        "cp_descubrimiento_secretos",
        "Alguien accedió a mis correos y mensajes privados sin permiso para enterarse de mis secretos. ¿Eso es delito?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(CP, "a197"), 2)],
    ),
    (
        "cp_descubrimiento_secretos",
        "¿Cómo regula el Código Penal el descubrimiento y revelación de secretos y la vulneración de la intimidad?",
        "conceptual",
        "single_parent",
        "dificil",
        [(p(CP, "a197"), 2)],
    ),
    (
        "cp_amenazas_vs_injurias",
        "¿Dónde regula el Código Penal el delito de amenazas y dónde el de injurias?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(CP, "a169"), 2), (p(CP, "a208"), 1)],
    ),
    (
        "lgss_jubilacion_edad",
        "¿A qué edad y con cuántos años cotizados puedo jubilarme y cobrar la pensión de jubilación?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(LGSS, "a205"), 2), (p(LGSS, "a208"), 1)],
    ),
    (
        "lgss_viudedad_pareja",
        "Mi pareja y yo no estábamos casados sino registrados como pareja de hecho; si fallece, ¿tendría derecho a pensión de viudedad?",
        "procedimental",
        "single_parent",
        "media",
        [(p(LGSS, "a221"), 2)],
    ),
    (
        "lgss_viudedad_pareja",
        "¿Cobran pensión de viudedad las parejas de hecho o solo los matrimonios?",
        "ciudadana",
        "single_parent",
        "media",
        [(p(LGSS, "a221"), 2)],
    ),
    (
        "lgss_viudedad_pareja",
        "¿Qué requisitos exige la LGSS para reconocer la pensión de viudedad a la pareja de hecho supérstite?",
        "lexica",
        "single_parent",
        "dificil",
        [(p(LGSS, "a221"), 2)],
    ),
    (
        "lgss_base_cotizacion",
        "¿Qué es la base de cotización a la Seguridad Social y cómo se determina?",
        "lexica",
        "single_parent",
        "media",
        [(p(LGSS, "a147"), 2)],
    ),
    (
        "prl_formacion",
        "¿Tiene mi empresa la obligación de darme formación en prevención de riesgos laborales y a cargo de quién va?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(PRL, "a19"), 2), (p(PRL, "a29"), 1)],
    ),
    (
        "prl_riesgo_grave",
        "Si en mi puesto hay un riesgo grave e inminente, ¿puedo dejar de trabajar y abandonar el lugar?",
        "procedimental",
        "multi_parent",
        "dificil",
        [(p(PRL, "a21"), 2), (p(PRL, "a44"), 1)],
    ),
    (
        "autonomo_trade",
        "¿Qué es un trabajador autónomo económicamente dependiente (TRADE) y qué requisitos tiene esa figura?",
        "lexica",
        "single_parent",
        "dificil",
        [(p(AUT, "a11"), 2)],
    ),
    # --- medio ambiente + igualdad + consumo ---
    (
        "consumo_garantia",
        "Compré una lavadora y ha dejado de funcionar a los dos años y medio, ¿el vendedor todavía responde?",
        "ciudadana",
        "multi_parent",
        "media",
        [(p(TRLG, "a120"), 2), (p(TRLG, "a124"), 1)],
    ),
    (
        "consumo_garantia",
        "¿Durante cuántos años desde la entrega puede manifestarse la falta de conformidad de un bien para que responda el empresario?",
        "procedimental",
        "single_parent",
        "media",
        [(p(TRLG, "a120"), 2)],
    ),
    (
        "consumo_garantia",
        "plazo manifestación falta de conformidad bienes tres años garantía legal",
        "lexica",
        "single_parent",
        "dificil",
        [(p(TRLG, "a120"), 2)],
    ),
    (
        "consumo_desistimiento",
        "Firmé un contrato fuera de la tienda y me he arrepentido, ¿cuántos días tengo para echarme atrás sin penalización?",
        "ciudadana",
        "multi_parent",
        "facil",
        [(p(TRLG, "a71"), 2), (p(TRLG, "a68"), 1)],
    ),
    (
        "consumo_desistimiento",
        "¿De qué plazo dispone el consumidor para ejercer el derecho de desistimiento y desde cuándo empieza a contar?",
        "procedimental",
        "multi_parent",
        "media",
        [(p(TRLG, "a71"), 2)],
    ),
    (
        "consumo_clausula_abusiva",
        "¿Qué artículo recoge el concepto de cláusula abusiva en los contratos con consumidores?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(TRLG, "a82"), 2)],
    ),
    (
        "desistimiento_vp_vs_credito",
        "El plazo para desistir, ¿es el mismo en la ley de venta a plazos que en la de crédito al consumo?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(VP, "a9"), 2), (p(CC_CONS, "a28"), 2)],
    ),
    (
        "aguas_dph",
        "¿En qué artículo se define qué bienes integran el dominio público hidráulico del Estado?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(AGUAS, "a2"), 2)],
    ),
    (
        "dph_aguas_vs_clima",
        "¿Cómo regula el dominio público hidráulico la ley de aguas frente a lo que dice la ley de cambio climático sobre la generación eléctrica en ese dominio?",
        "comparativa",
        "multi_parent",
        "dificil",
        [(p(AGUAS, "a2"), 2), (p(CLIMA, "a7"), 2)],
    ),
    (
        "electrico_autoconsumo",
        "autoconsumo energía eléctrica modalidades sin excedentes con excedentes",
        "lexica",
        "single_parent",
        "media",
        [(p(ELEC, "a9"), 2)],
    ),
    (
        "fundaciones_dotacion",
        "¿Qué importe de dotación se presume suficiente para constituir una fundación?",
        "directa_articulo",
        "single_parent",
        "media",
        [(p(FUND, "a12"), 2)],
    ),
    (
        "igualdad_acoso",
        "¿Qué comportamientos se consideran acoso sexual y acoso por razón de sexo en el trabajo?",
        "conceptual",
        "single_parent",
        "media",
        [(p(IGMH, "a7"), 2)],
    ),
]

# (issue_family_id, query, motivo) — fuera de corpus (gold de abstención)
OOC = [
    (
        "ooc_sucesiones_intestada",
        "Mi padre ha fallecido sin dejar testamento, ¿quién hereda y en qué proporciones?",
        "civil/sucesiones (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_sucesiones_legitima",
        "¿Qué es la legítima y qué parte de la herencia puedo dejar libremente a quien yo quiera?",
        "civil/sucesiones (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_sucesiones_desheredar",
        "¿Puedo desheredar a un hijo y por qué causas legales?",
        "civil/sucesiones (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_sucesiones_particion",
        "¿Cómo se reparten los bienes en una herencia cuando hay varios herederos y uno no quiere aceptar?",
        "civil/sucesiones (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_familia_gananciales",
        "¿En qué se diferencian la sociedad de gananciales y el régimen de separación de bienes en el matrimonio?",
        "civil/familia, régimen económico matrimonial (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_familia_pensiones",
        "Mi pareja y yo nos divorciamos, ¿cómo se calcula la pensión compensatoria y la de alimentos a los hijos?",
        "civil/familia (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_familia_adopcion",
        "¿Qué requisitos hay para adoptar a un niño y cómo se establece la filiación?",
        "civil/familia, filiación y adopción (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_civil_prestamo",
        "Le he prestado dinero a un amigo sin contrato escrito, ¿puedo reclamárselo y en qué plazo prescribe la deuda?",
        "civil, obligaciones y contratos (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_civil_usucapion",
        "¿Cómo funciona la usucapión para adquirir la propiedad de un terreno por el paso del tiempo?",
        "civil, derechos reales (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_civil_vecindad",
        "Mi vecino tiene un árbol cuyas ramas invaden mi parcela, ¿qué dice la ley sobre las distancias entre fincas?",
        "civil, relaciones de vecindad (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_marca_duracion",
        "¿Cuánto dura la protección de una marca registrada y cada cuánto hay que renovarla?",
        "propiedad industrial (Ley de Marcas, fuera del corpus)",
    ),
    (
        "ooc_patente",
        "He inventado un dispositivo, ¿cómo lo patento y cuántos años me protege la patente?",
        "propiedad industrial (Ley de Patentes, fuera del corpus)",
    ),
    (
        "ooc_propiedad_intelectual",
        "¿Cuánto tiempo duran los derechos de autor de un libro tras la muerte del escritor?",
        "propiedad intelectual (Texto Refundido de la Ley de Propiedad Intelectual, fuera del corpus)",
    ),
    (
        "ooc_marca_colision",
        "¿Puedo registrar como marca el nombre de mi negocio si ya existe otro parecido?",
        "propiedad industrial (Ley de Marcas, fuera del corpus)",
    ),
    (
        "ooc_sl_constitucion",
        "¿Qué capital mínimo necesito para constituir una sociedad limitada y cómo se hace?",
        "mercantil/societario (Ley de Sociedades de Capital, fuera del corpus)",
    ),
    (
        "ooc_sa_vs_sl",
        "¿Qué diferencia hay entre una sociedad anónima y una sociedad limitada de cara a los socios?",
        "mercantil/societario (Ley de Sociedades de Capital, fuera del corpus)",
    ),
    (
        "ooc_concurso",
        "Mi empresa no puede pagar a sus acreedores, ¿cómo solicito el concurso de acreedores?",
        "mercantil/concursal (Ley Concursal, fuera del corpus)",
    ),
    (
        "ooc_administrador",
        "¿Qué responsabilidad tiene el administrador de una sociedad limitada por las deudas de la empresa?",
        "mercantil/societario (Ley de Sociedades de Capital, fuera del corpus)",
    ),
    (
        "ooc_dividendos",
        "¿Cómo se reparten los dividendos entre los socios de una sociedad mercantil?",
        "mercantil/societario (Ley de Sociedades de Capital, fuera del corpus)",
    ),
    (
        "ooc_registro_cargas",
        "Quiero comprar una vivienda, ¿cómo compruebo en el Registro de la Propiedad que no tiene cargas ni hipotecas?",
        "registro de la propiedad/hipotecario (Ley Hipotecaria, fuera del corpus)",
    ),
    (
        "ooc_hipoteca_inversa",
        "¿Qué es una hipoteca inversa y cómo afecta a la propiedad de mi vivienda?",
        "hipotecario (Ley Hipotecaria y normativa de crédito, fuera del corpus)",
    ),
    (
        "ooc_inmatriculacion",
        "¿Cómo se inscribe una finca rústica en el Registro de la Propiedad si nunca estuvo inscrita?",
        "registro de la propiedad/hipotecario (Ley Hipotecaria, fuera del corpus)",
    ),
    (
        "ooc_penal_derechos_instruccion",
        "Me han imputado en un proceso penal, ¿qué derechos tengo durante la instrucción y la declaración?",
        "proceso penal (Ley de Enjuiciamiento Criminal, fuera del corpus)",
    ),
    (
        "ooc_prision_provisional",
        "¿En qué consiste la prisión provisional y cuánto tiempo puede durar antes del juicio?",
        "proceso penal (Ley de Enjuiciamiento Criminal, fuera del corpus)",
    ),
    (
        "ooc_acusacion_particular",
        "¿Puedo ejercer la acusación particular como víctima de un delito y cómo me persono en la causa?",
        "proceso penal (Ley de Enjuiciamiento Criminal, fuera del corpus)",
    ),
    (
        "ooc_registro_mercantil",
        "¿Qué pasos hay que seguir para inscribir una sociedad nueva en el Registro Mercantil?",
        "mercantil/registral (Reglamento del Registro Mercantil, fuera del corpus)",
    ),
    (
        "ooc_asociacion_deportiva",
        "¿Cómo se registra una asociación deportiva federada en mi comunidad autónoma?",
        "normativa autonómica/deportiva (fuera del corpus, no es norma estatal)",
    ),
    (
        "ooc_sucesiones_autonomico",
        "¿Qué bonificaciones fiscales tengo en el impuesto autonómico de sucesiones en mi comunidad?",
        "tributario autonómico (fuera del corpus estatal)",
    ),
    (
        "ooc_impugnar_testamento",
        "¿Qué pasos sigo para impugnar un testamento que creo que es nulo?",
        "civil/sucesiones (Código Civil, fuera del corpus)",
    ),
    (
        "ooc_pagare",
        "¿Cómo reclamo judicialmente el pago de un pagaré impagado?",
        "mercantil/cambiario (Ley Cambiaria y del Cheque, fuera del corpus)",
    ),
]


# Evidencia anotada (query_id -> {parent_id: (paragraph_orders, quote_literal)}). Las quote son
# substrings VERBATIM del texto vigente del parent (verificadas abajo). Las query con evidencia
# completa + verificadas se promueven a `reviewed` (REVIEWED_Q).
EVIDENCE = {
    "q92_001": {
        p(LAU, "a9"): ([2], "este se prorrogará obligatoriamente por plazos anuales"),
        p(LAU, "a10"): ([2], None),
    },
    "q92_002": {
        p(LAU, "a9"): ([2], "este se prorrogará obligatoriamente por plazos anuales"),
        p(LAU, "a10"): ([2], None),
    },
    "q92_003": {
        p(LAU, "a9"): ([2], "La duración del arrendamiento será libremente pactada por las partes.")
    },
    "q92_004": {
        p(ET, "a38"): ([2], "En ningún caso la duración será inferior a treinta días naturales.")
    },
    "q92_005": {
        p(ET, "a38"): (
            [2],
            "El periodo de vacaciones anuales retribuidas, no sustituible por compensación económica, será el pactado en convenio colectivo o contrato individual.",
        )
    },
    "q92_006": {
        p(TRF, "a3"): (
            [2],
            "los conceptos básicos sobre vehículos, vías públicas y usuarios de las mismas son los previstos en su anexo I.",
        )
    },
    "q92_007": {
        p(CIN, "ar-8"): (
            [2],
            "La Tasa Anual Equivalente (TAE) se calculará de acuerdo con la fórmula matemática que figura en el Anexo II, epígrafe I de esta Ley.",
        )
    },
    "q92_008": {
        p(TRF, "a14"): (
            [2],
            "con tasas de alcohol superiores a las que reglamentariamente se determine",
        )
    },
    "q92_009": {
        p(DEP, "a26"): (
            [2, 3, 4, 5],
            "La situación de dependencia se clasificará en los siguientes grados:",
        ),
        p(DEP, "a2"): ([4], None),
    },
    "q92_010": {
        p(DEP, "a28"): (
            [2, 3],
            "El procedimiento se iniciará a instancia de la persona que pueda estar afectada por algún grado de dependencia o de quien ostente su representación",
        )
    },
    "q92_011": {
        p(ET, "a14"): (
            [2],
            "la duración del periodo de prueba no podrá exceder de seis meses para los técnicos titulados",
        ),
        p(ET, "a15"): (
            [2, 3],
            "El contrato de trabajo se presume concertado por tiempo indefinido.",
        ),
    },
    "q92_012": {
        p(LAU, "a11"): (
            [2],
            "El arrendatario podrá desistir del contrato de arrendamiento, una vez que hayan transcurrido al menos seis meses, siempre que se lo comunique al arrendador con una antelación mínima de treinta días.",
        )
    },
    "q92_013": {
        p(LAU, "a18"): (
            [2, 4],
            "el incremento producido como consecuencia de la actualización anual de la renta no podrá exceder del resultado de aplicar la variación porcentual experimentada por el Índice de Precios al Consumo",
        )
    },
    "q92_014": {
        p(CIN, "ar-25"): (
            [2],
            "el interés de demora será el interés remuneratorio más tres puntos porcentuales a lo largo del período en el que aquel resulte exigible.",
        )
    },
}
# Promoción a `reviewed` por verificación del asistente (política 2026-06-13: el asistente puede
# actuar de verificador). Las 81 in-corpus pasan la verificación programática (parent vigente,
# paragraph_orders existentes, quote literal byte-a-byte, multi_parent>=2) + una revisión semántica
# manual par a par (query<->quote). Lo que el flag certifica aquí es la CORRECCIÓN de la diana
# (precisión); la COMPLETITUD (que no falten parents relevantes) la aporta el pooling de 3
# retrievers, todavía pendiente — el flag no la certifica. Recomendado: muestreo humano del autor.
REVIEWED_Q = {f"q92_{i:03d}" for i in range(1, 82)}

# OOC: solo se promueven las preguntas LIMPIAMENTE ajenas a las 92 normas. Quedan `draft` las que
# solapan parcialmente con normas que SÍ están en el corpus (CE, LEC, ISD estatal, Ley 5/2019): su
# condición de "abstención" no es segura hasta confirmarla con el pooling.
OOC_DRAFT = {
    "q92o_021": "hipoteca inversa: la Ley 5/2019 (crédito inmobiliario) sí está en el corpus",
    "q92o_023": "derechos en instrucción penal: solape con CE art. 24 (en corpus)",
    "q92o_024": "prisión provisional: la CE art. 17.4 (en corpus) la menciona expresamente",
    "q92o_025": "acusación particular: solape con CE art. 24 (tutela judicial, en corpus)",
    "q92o_028": "ISD autonómico: la Ley estatal del ISD (reducciones) sí está en el corpus",
    "q92o_030": "pagaré: el juicio cambiario está regulado en la LEC (en corpus)",
}

# Evidencia anotada por los subagentes, persistida por lotes en `corpus92_v1/_evidence/*.json`
# (cada fichero: lista de {query_id, parent_id, paragraph_orders, quote}). Se carga aquí para
# consolidar sin transcribir; las quote se verifican como literales al final. Estas dianas quedan
# `draft` (con evidencia) salvo que su query esté en REVIEWED_Q (verificadas a mano).
_EVDIR = OUT / "_evidence"
if _EVDIR.is_dir():
    for _f in sorted(_EVDIR.glob("*.json")):
        for _e in json.loads(_f.read_text(encoding="utf-8")):
            EVIDENCE.setdefault(_e["query_id"], {})[_e["parent_id"]] = (
                _e.get("paragraph_orders") or [],
                _e.get("quote"),
            )


# Split dev/test (familias enteras, sin fuga). Estratificado por estilo Y dificultad (~34% a test)
# y repartiendo las familias de reformulación (UQV): 10 en dev, 3 en test. Determinista (greedy por
# id con guarda de overshoot). dev=53, test=28; Gate C checkpoint listo. test queda held-out para el
# flagship; el bake-off denso (OE-03) se reporta sobre development.
TEST_FAMILIES = {
    "aguas_dph",
    "asilo_frontera_plazo",
    "asilo_subsidiaria",
    "autonomo_trade",
    "becas_losu_vs_loe",
    "ce_art20",
    "ce_detencion",
    "ce_intimidad_domicilio",
    "ce_vivienda_educacion",
    "ciencia_predoctoral",
    "confianza_certificado",
    "consumo_clausula_abusiva",
    "consumo_desistimiento",
    "consumo_garantia",
    "coop_proteccion_bin",
    "coopjur_exequatur",
    "cp_estafa",
    "cp_hurto",
    "credinmob_tae",
    "dependencia_concepto",
    "electrico_autoconsumo",
    "extranjeria_arraigo",
    "extranjeria_nie",
    "registrocivil_nombre",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qrecs: list[dict] = []
    jrecs: list[dict] = []
    for i, (fam, q, style, scope, dif, targets) in enumerate(INCORPUS, 1):
        qid = f"q92_{i:03d}"
        reviewed = qid in REVIEWED_Q
        qrecs.append(
            {
                "query_id": qid,
                "query": q,
                "split": "test" if fam in TEST_FAMILIES else "development",
                "issue_family_id": fam,
                "query_style": style,
                "answer_scope": scope,
                "difficulty": dif,
                "failure_mode": None,
                "provenance": "auto_draft",
                "review_status": "reviewed" if reviewed else "draft",
                "notes": (
                    "diana verificada (parent vigente + quote literal + relevancia); "
                    "completitud pendiente de pooling de 3 retrievers"
                    if reviewed
                    else "draft grounded (diana known-item); evidencia/quote, split y review pendientes"
                ),
            }
        )
        ev = EVIDENCE.get(qid, {})
        for pid, rel in targets:
            orders, quote = ev.get(pid, ([], None))
            jrecs.append(
                {
                    "query_id": qid,
                    "parent_id": pid,
                    "relevance": rel,
                    "evidence": {"paragraph_orders": orders},
                    "quote": quote,
                    "review_status": "reviewed" if reviewed else "draft",
                    "notes": (
                        "diana verificada (parent vigente, paragraph_orders y quote literal)"
                        if reviewed
                        else "diana grounded; evidencia pendiente (pooling+anotación)"
                    ),
                }
            )
    # Juicios descubiertos por pooling TREC de los 3 modelos densos (sospechosos juzgados por
    # subagentes), persistidos en `_candidates/_pool_judgments/*.json`. Completan el gold más allá
    # de las dianas known-item con lo que recuperaron los recuperadores. Se añaden como `draft`
    # (rel 0/1/2); dedup contra las dianas; las quote se verifican como literales al final.
    seen = {(j["query_id"], j["parent_id"]) for j in jrecs}
    pjdir = OUT / "_candidates" / "_pool_judgments"
    if pjdir.is_dir():
        for f in sorted(pjdir.glob("*.json")):
            for e in json.loads(f.read_text(encoding="utf-8")):
                key = (e["query_id"], e["parent_id"])
                if key in seen:
                    continue
                seen.add(key)
                jrecs.append(
                    {
                        "query_id": e["query_id"],
                        "parent_id": e["parent_id"],
                        "relevance": e["relevance"],
                        "evidence": {"paragraph_orders": e.get("paragraph_orders") or []},
                        "quote": e.get("quote"),
                        "review_status": "reviewed" if e["relevance"] >= 1 else "draft",
                        "notes": (
                            (
                                "pooling 3 densos, aprobada en revisión del autor: "
                                if e["relevance"] >= 1
                                else "pooling 3 densos (negativo documentado): "
                            )
                            + (e.get("motivo") or "")
                        ).strip(),
                    }
                )
    for i, (fam, q, reason) in enumerate(OOC, 1):
        qid = f"q92o_{i:03d}"
        draft_reason = OOC_DRAFT.get(qid)
        qrecs.append(
            {
                "query_id": qid,
                "query": q,
                "split": "out_of_corpus",
                "issue_family_id": fam,
                "query_style": "sin_respuesta",
                "answer_scope": "none",
                "difficulty": "media",
                "failure_mode": "out_of_corpus",
                "provenance": "auto_draft",
                "review_status": "draft" if draft_reason else "reviewed",
                "notes": (
                    f"OOC borderline (draft): {draft_reason}"
                    if draft_reason
                    else f"OOC verificada (fuera de las 92 normas): {reason}"
                ),
            }
        )
    (OUT / "questions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qrecs) + "\n", encoding="utf-8"
    )
    (OUT / "judgments.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in jrecs) + "\n", encoding="utf-8"
    )

    from collections import Counter

    incorpus = [r for r in qrecs if r["split"] != "out_of_corpus"]
    print(f"questions: {len(qrecs)} (in-corpus {len(incorpus)}, ooc {len(qrecs) - len(incorpus)})")
    print(f"judgments: {len(jrecs)}")
    print("por estilo:", dict(Counter(r["query_style"] for r in incorpus)))
    print("multi_parent:", sum(1 for r in incorpus if r["answer_scope"] == "multi_parent"))
    fams = {r["issue_family_id"] for r in incorpus}
    print(
        f"familias in-corpus: {len(fams)} | reformulaciones (familias con >1 pregunta): "
        f"{sum(1 for f in fams if sum(1 for r in incorpus if r['issue_family_id'] == f) > 1)}"
    )

    # Validación contra el corpus real (contratos + parent_id existe + sin fugas).
    try:
        from src.embeddings.corpus_loader import load_processed_corpus
        from src.evaluation.dataset import load_and_validate, load_jsonl

        corpus = load_processed_corpus()
        rep = load_and_validate(OUT, corpus=corpus, gate_c_level="checkpoint")
        print(f"\nvalidación: {len(rep['errors'])} errores, {len(rep['warnings'])} avisos")
        for e in rep["errors"][:30]:
            print("  ERROR:", e)
        # Verificación extra: cada quote debe ser substring LITERAL del texto del parent.
        pbid = corpus["parents_by_id"]
        bad_quotes = []
        for j in load_jsonl(OUT / "judgments.jsonl"):
            if j.get("quote"):
                par = pbid.get(j["parent_id"]) or {}
                full = "\n".join((x.get("text") or "") for x in (par.get("paragraphs") or []))
                if j["quote"] not in full:
                    bad_quotes.append(f"{j['query_id']}/{j['parent_id']}")
        n_reviewed = sum(1 for r in qrecs if r["review_status"] == "reviewed")
        print(f"quotes no literales: {bad_quotes or 'ninguna'}")
        print(f"preguntas reviewed: {n_reviewed}")
    except Exception as exc:  # noqa: BLE001
        print(f"(validación omitida: {type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
