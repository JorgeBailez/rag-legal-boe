"""Convierte el scaffold de anotación del juez a/desde un Markdown cómodo de rellenar a mano.

El scaffold (`validate_judge.py --scaffold`) es JSONL: incómodo de leer y anotar. Este script:

  to-md    --scaffold X.jsonl --out worksheet.md
      Genera una hoja legible (una pregunta por bloque) con pregunta + respuesta + referencia +
      evidencia, y un hueco para tu anotación. **OCULTA el veredicto del juez** para forzar la
      anotación a ciegas (anti-anclaje).

  from-md  --md worksheet.md --out annotations.jsonl
      Vuelca tu anotación (FIDELIDAD / CORRECCION / NOTAS) al JSONL que consume
      `validate_judge.py --annotations`.

Flujo: scaffold → to-md → (anotas la .md) → from-md → validate_judge --annotations. Sin red.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HEADER = """# Hoja de anotación — validación del juez (corpus92)

Rellena SOLO las tres líneas bajo `>>> TU ANOTACIÓN <<<` de cada pregunta. No toques las cabeceras
`## <id>`.

- **FIDELIDAD**: `true` si CADA afirmación de la respuesta está soportada por la EVIDENCIA; `false`
  si al menos una no se deduce de ella (grounding, no verdad real; el aviso legal no cuenta).
- **CORRECCION**: `correct` | `partial` | `incorrect` comparando contra la REFERENCIA. Paráfrasis
  fiel = `correct`. Si no hay referencia, déjalo vacío.
- **NOTAS**: opcional (dudas, justificación).

---
"""

_BLOCK = """\
## {query_id}

**PREGUNTA**
{question}

**RESPUESTA GENERADA (lo que evalúas)**
{answer}

**REFERENCIA (gold — verdad para CORRECCIÓN){ref_note}**
{reference}

**EVIDENCIA (verdad para FIDELIDAD)**
{evidence}

**>>> TU ANOTACIÓN <<<**
FIDELIDAD: {faith}
CORRECCION: {corr}
NOTAS: {notes}

---
"""


def to_md(rows: list[dict]) -> str:
    """Scaffold JSONL → Markdown legible (sin mostrar el veredicto del juez)."""
    parts = [_HEADER]
    for r in rows:
        ref = (r.get("reference_answer") or "").strip()
        parts.append(
            _BLOCK.format(
                query_id=r.get("query_id", ""),
                question=(r.get("question") or "").strip(),
                answer=(r.get("answer_text") or "").strip(),
                ref_note="" if ref else " — VACÍA: no anotes corrección",
                reference=ref or "(sin referencia)",
                evidence=(r.get("evidences_block") or "(sin evidencia)").strip(),
                faith=_bool_to_text(r.get("human_faithful")),
                corr=r.get("human_correctness") or "",
                notes=r.get("notes") or "",
            )
        )
    return "\n".join(parts)


def _bool_to_text(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


_TRUE = {"true", "verdadero", "si", "sí", "1", "fiel"}
_FALSE = {"false", "falso", "no", "0", "infiel"}
_CORR = {"correct", "partial", "incorrect"}


def _parse_faith(value: str) -> bool | None:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def _parse_corr(value: str) -> str:
    v = value.strip().lower()
    return v if v in _CORR else ""


def from_md(text: str) -> tuple[list[dict], list[str]]:
    """Markdown rellenado → (filas de anotación, avisos). Una fila por bloque `## <id>`."""
    blocks = re.split(r"(?m)^##\s+", text)
    rows: list[dict] = []
    warnings: list[str] = []
    for block in blocks[1:]:  # blocks[0] es el encabezado previo al primer '##'
        lines = block.splitlines()
        query_id = lines[0].strip().split()[0] if lines and lines[0].strip() else ""
        if not query_id:
            continue
        faith_raw = _field(block, "FIDELIDAD")
        corr_raw = _field(block, "CORRECCION")
        notes = _field(block, "NOTAS")
        faith = _parse_faith(faith_raw) if faith_raw is not None else None
        corr = _parse_corr(corr_raw) if corr_raw is not None else ""
        if faith_raw and faith_raw.strip() and faith is None:
            warnings.append(
                f"{query_id}: FIDELIDAD no reconocida ({faith_raw.strip()!r}); se deja sin anotar."
            )
        if corr_raw and corr_raw.strip() and corr == "":
            warnings.append(
                f"{query_id}: CORRECCION no reconocida ({corr_raw.strip()!r}); se deja vacía."
            )
        rows.append(
            {
                "query_id": query_id,
                "human_faithful": faith,
                "human_correctness": corr,
                "notes": (notes or "").strip(),
            }
        )
    return rows, warnings


def _field(block: str, key: str) -> str | None:
    """Devuelve el texto tras `KEY:` en una línea del bloque (None si la línea no aparece)."""
    m = re.search(rf"(?mi)^\s*{key}\s*:(.*)$", block)
    return m.group(1) if m else None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Conversor scaffold/Markdown de anotación.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_to = sub.add_parser("to-md", help="scaffold JSONL → worksheet Markdown")
    p_to.add_argument("--scaffold", required=True)
    p_to.add_argument("--out", required=True)
    p_from = sub.add_parser("from-md", help="worksheet Markdown rellenado → annotations JSONL")
    p_from.add_argument("--md", required=True)
    p_from.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.cmd == "to-md":
        rows = [
            json.loads(line)
            for line in Path(args.scaffold).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        Path(args.out).write_text(to_md(rows), encoding="utf-8")
        print(f"Hoja escrita: {args.out} ({len(rows)} preguntas). Rellena FIDELIDAD/CORRECCION.")
        return 0

    rows, warnings = from_md(Path(args.md).read_text(encoding="utf-8"))
    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    n_f = sum(1 for r in rows if r["human_faithful"] is not None)
    n_c = sum(1 for r in rows if r["human_correctness"])
    print(
        f"Anotación escrita: {args.out} ({len(rows)} filas; {n_f} con fidelidad, {n_c} corrección)."
    )
    for w in warnings:
        print(f"  ⚠ {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
