"""Construcción del prompt restrictivo para la generación fundamentada.

Carga las plantillas de `prompts/` (rutas relativas al repositorio, nunca absolutas del portátil)
y construye los mensajes `system`/`user` para Ollama:

- las evidencias se delimitan e identifican con IDs compactos (`E1`, `E2`, ...);
- evidencias y pregunta se marcan explícitamente como DATOS no confiables, no instrucciones;
- se obliga a responder SOLO con el contexto y a citar exclusivamente IDs entregados;
- se incrusta el JSON Schema serializado (además de enviarse en `format` por el cliente).

El reemplazo de placeholders es **literal y en una sola pasada** (`re.sub`): el texto jurídico
puede contener llaves `{}` sin romper el renderizado, porque el contenido inyectado no se vuelve a
escanear y no se interpreta como placeholder.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.contracts.generation_models import RagLlmAnswerV1
from src.retrieval.evidence_builder import GenerationEvidence

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
RAG_PROMPT_FILE = "rag_prompt.txt"

# Placeholders admitidos en la plantilla del prompt de usuario (reemplazo literal, no str.format).
_KNOWN_PLACEHOLDERS = ("schema", "allowed_ids", "evidences", "question")
_PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(_KNOWN_PLACEHOLDERS) + r")\}")


@lru_cache(maxsize=8)
def load_template(name: str, prompts_dir: str | None = None) -> str:
    """Carga una plantilla de `prompts/` por nombre de fichero (cacheada)."""
    base = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    return (base / name).read_text(encoding="utf-8")


def render_template(template: str, mapping: dict[str, str]) -> str:
    """Sustituye los placeholders conocidos en una sola pasada (seguro frente a llaves en datos)."""

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in mapping:
            raise KeyError(f"placeholder sin valor: {key!r}")
        return mapping[key]

    return _PLACEHOLDER_RE.sub(_repl, template)


def serialize_llm_schema(schema: dict | None = None) -> str:
    """Serializa el JSON Schema del contrato del LLM de forma determinista (para el prompt)."""
    schema = schema if schema is not None else RagLlmAnswerV1.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)


def build_evidences_block(evidences: list[GenerationEvidence]) -> str:
    """Bloque de evidencias delimitado, una por ID, con su etiqueta y su texto jurídico íntegro."""
    parts: list[str] = []
    for ev in evidences:
        parts.append(
            f"--- {ev.evidence_id} | {ev.label} ---\n{ev.text}\n--- fin {ev.evidence_id} ---"
        )
    return "\n\n".join(parts)


def allowed_ids_str(evidences: list[GenerationEvidence]) -> str:
    """Lista legible de IDs permitidos para `citation_ids` (p. ej. 'E1, E2, E3')."""
    return ", ".join(ev.evidence_id for ev in evidences)


def build_user_prompt(
    *,
    question: str,
    evidences: list[GenerationEvidence],
    llm_schema: dict | None = None,
    prompts_dir: str | None = None,
) -> str:
    """Renderiza el prompt de usuario (evidencias + pregunta + schema + IDs permitidos)."""
    template = load_template(RAG_PROMPT_FILE, prompts_dir)
    rendered = render_template(
        template,
        {
            "schema": serialize_llm_schema(llm_schema),
            "allowed_ids": allowed_ids_str(evidences) or "(ninguno)",
            "evidences": build_evidences_block(evidences),
            "question": question,
        },
    )
    if _PLACEHOLDER_RE.search(rendered) and not evidences:
        # Solo posible si la plantilla quedara incompleta; se trata como error de plantilla.
        raise KeyError("placeholders residuales en el prompt renderizado")
    return rendered


def build_messages(
    *,
    question: str,
    evidences: list[GenerationEvidence],
    llm_schema: dict | None = None,
    prompts_dir: str | None = None,
) -> list[dict[str, str]]:
    """Construye los mensajes `system`/`user` para `/api/chat` de Ollama."""
    system = load_template(SYSTEM_PROMPT_FILE, prompts_dir)
    user = build_user_prompt(
        question=question,
        evidences=evidences,
        llm_schema=llm_schema,
        prompts_dir=prompts_dir,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
