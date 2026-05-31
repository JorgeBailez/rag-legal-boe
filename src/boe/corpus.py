"""Corpus semilla del MVP: carga y verificación de normas contra el raw descargado.

La verificación se hace **sin red**, a partir del raw ya descargado por `BoeClient`:
se evalúan existencia, metadatos clave, disponibilidad de endpoints y los criterios de
inclusión (vigente + estado de consolidación "Finalizado").
"""

from __future__ import annotations

import json
from pathlib import Path

from src.boe.parser import load_xml, parse_metadata, validate_response

SEED_CORPUS_PATH = Path("data/corpus/seed_corpus.json")

# Endpoints (nombres lógicos de `BoeClient.ENDPOINTS`) obligatorios y opcionales.
MANDATORY_ENDPOINTS = ("metadatos", "texto", "indice")
OPTIONAL_ENDPOINTS = ("analisis", "metadata_eli", "full")

FINALIZADO_LABEL = "Finalizado"


def load_seed_corpus(path: Path = SEED_CORPUS_PATH) -> list[dict]:
    """Carga la lista de normas del corpus semilla."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["norms"]


def evaluate_criteria(metadata: dict, availability: dict[str, bool]) -> dict:
    """Evalúa los criterios de inclusión a partir de metadatos y disponibilidad.

    Criterio: vigente (no derogada y vigencia no agotada) + estado_consolidacion
    "Finalizado" + endpoints obligatorios disponibles.
    """
    derogation = metadata.get("derogation_status")
    expired = metadata.get("expired_validity")
    consolidation = (metadata.get("consolidation_status") or {}).get("label")

    vigente = derogation == "N" and expired == "N"
    finalizado = consolidation == FINALIZADO_LABEL
    mandatory_available = all(availability.get(e, False) for e in MANDATORY_ENDPOINTS)

    reasons: list[str] = []
    if not vigente:
        reasons.append("no vigente (derogada o vigencia agotada)")
    if not finalizado:
        reasons.append(f"estado_consolidacion != {FINALIZADO_LABEL} ({consolidation!r})")
    if not mandatory_available:
        missing = [e for e in MANDATORY_ENDPOINTS if not availability.get(e, False)]
        reasons.append(f"faltan endpoints obligatorios: {missing}")

    return {
        "meets_criteria": vigente and finalizado and mandatory_available,
        "vigente": vigente,
        "finalizado": finalizado,
        "reasons": reasons,
    }


def verify_norm(norm_id: str, raw_dir: Path, downloaded: dict[str, Path]) -> dict:
    """Construye la fila de verificación de una norma a partir de su raw descargado.

    `downloaded` mapea nombre de endpoint -> ruta guardada (lo que devuelve
    `BoeClient.download_norm_raw`). No hace red.
    """
    raw_dir = Path(raw_dir)
    availability = {e: e in downloaded for e in (*MANDATORY_ENDPOINTS, *OPTIONAL_ENDPOINTS)}

    metadatos_path = raw_dir / norm_id / "metadatos.xml"
    if not metadatos_path.is_file():
        return {
            "norm_id": norm_id,
            "exists": False,
            "availability": availability,
            "meets_criteria": False,
            "reasons": ["metadatos no disponible"],
        }

    data = validate_response(load_xml(metadatos_path), metadatos_path)
    metadata = parse_metadata(data)
    criteria = evaluate_criteria(metadata, availability)

    return {
        "norm_id": norm_id,
        "exists": True,
        "title": metadata.get("title"),
        "rank": metadata.get("rank"),
        "scope": metadata.get("scope"),
        "consolidation_status": metadata.get("consolidation_status"),
        "expired_validity": metadata.get("expired_validity"),
        "derogation_status": metadata.get("derogation_status"),
        "effective_date": metadata.get("effective_date"),
        "availability": availability,
        "vigente": criteria["vigente"],
        "finalizado": criteria["finalizado"],
        "meets_criteria": criteria["meets_criteria"],
        "reasons": criteria["reasons"],
    }
