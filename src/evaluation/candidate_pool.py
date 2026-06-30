"""Pooling de candidatos para anotar el gold de relevancia (lógica pura, sin torch ni disco).

Construye, por pregunta, el conjunto de parents candidatos a juzgar reuniéndolos desde VARIOS
sistemas de recuperación (*pooling* de TREC: evita que el gold favorezca al recuperador que ya
tienes, requisito para que la comparación denso/BM25/híbrido no esté sesgada de origen). Incluye
los parents ya juzgados aunque ningún sistema los recupere (señal de recall / gold dudoso). Genera
además un *worksheet* legible para que el anotador gradúe la relevancia y elija la evidencia.

Funciones puras: la orquestación (cargar bundles, recuperar con torch) vive en
`scripts/build_eval_candidates.py`. Aquí solo se manipulan dicts simples, testeable offline.
"""

from __future__ import annotations

_SENTINEL_RANK = 10**9

_PROTOCOL_HEADER = (
    "> **Protocolo (resumen).** rel **2** = central/suficiente · rel **1** = apoyo/matiz · "
    "rel **0** = descartado (negativo duro o trampa temporal «Sin contenido»). "
    "`evidence.paragraph_orders` = los `[order]` que sustentan la relevancia (obligatorio en "
    "rel=2 reviewed). `multi_parent` = ≥2 parents NECESARIOS. `answerable=false` en OOC / "
    "in-corpus-sin-respuesta. `reviewed` solo lo verificado al 100% contra el texto; lo dudoso, "
    "`draft`. Negativos: 1–2 tentadores por pregunta. Detalle: `docs/anotacion_gold_relevancia.md`."
)


def build_pool(
    systems: list[dict],
    judged_by_qid: dict[str, dict[str, dict]],
    question_ids: list[str],
) -> dict[str, dict]:
    """Reúne candidatos por pregunta desde varios sistemas + inyecta los ya juzgados no recuperados.

    `systems`: lista de
    `{"bundle_id", "query_profile_id", "hits_by_qid": {qid: [{"parent_id","rank","score"}, ...]}}`.
    `judged_by_qid`: `qid -> {parent_id -> judgment}` (con `relevance` y `review_status`).
    `question_ids`: todas las preguntas a incluir (las OOC sin hits también tienen entrada).

    Devuelve `qid -> {"candidates": [...], "judged_not_pooled": [parent_id, ...]}`. Cada candidato:
    `{parent_id, found_by, n_systems, best_rank, best_score, from_judgment, current_relevance,
    current_review_status}`. Orden estable: `n_systems` desc, `best_score` desc, `best_rank` asc,
    `parent_id` asc.
    """
    agg: dict[str, dict[str, dict]] = {qid: {} for qid in question_ids}
    for system in systems:
        bundle_id = system["bundle_id"]
        profile_id = system["query_profile_id"]
        for qid, hits in system.get("hits_by_qid", {}).items():
            if qid not in agg:
                continue
            # Dedup dentro del sistema: el mejor (menor) rank por parent.
            best_in_system: dict[str, dict] = {}
            for h in hits:
                pid = h["parent_id"]
                rank = int(h["rank"])
                prev = best_in_system.get(pid)
                if prev is None or rank < prev["rank"]:
                    best_in_system[pid] = {"rank": rank, "score": float(h["score"])}
            for pid, rs in best_in_system.items():
                cand = agg[qid].setdefault(pid, {"found_by": []})
                cand["found_by"].append(
                    {
                        "bundle_id": bundle_id,
                        "query_profile_id": profile_id,
                        "rank": rs["rank"],
                        "score": rs["score"],
                    }
                )

    pool: dict[str, dict] = {}
    for qid in question_ids:
        judged = judged_by_qid.get(qid, {})
        cands_by_pid = agg.get(qid, {})
        judged_not_pooled: list[str] = []
        for pid in judged:  # inyectar juzgados que ningún sistema recuperó
            if pid not in cands_by_pid:
                cands_by_pid[pid] = {"found_by": []}
                judged_not_pooled.append(pid)
        candidates: list[dict] = []
        for pid, cand in cands_by_pid.items():
            found_by = cand["found_by"]
            j = judged.get(pid)
            candidates.append(
                {
                    "parent_id": pid,
                    "found_by": found_by,
                    "n_systems": len(found_by),
                    "best_rank": min((f["rank"] for f in found_by), default=None),
                    "best_score": max((f["score"] for f in found_by), default=0.0),
                    "from_judgment": not found_by and j is not None,
                    "current_relevance": (j or {}).get("relevance"),
                    "current_review_status": (j or {}).get("review_status"),
                }
            )
        candidates.sort(
            key=lambda c: (
                -c["n_systems"],
                -c["best_score"],
                c["best_rank"] if c["best_rank"] is not None else _SENTINEL_RANK,
                c["parent_id"],
            )
        )
        pool[qid] = {"candidates": candidates, "judged_not_pooled": judged_not_pooled}
    return pool


def _found_by_compact(found_by: list[dict]) -> str:
    if not found_by:
        return "— (solo juzgado; ningún sistema lo recuperó)"
    parts = []
    for f in sorted(found_by, key=lambda x: x["rank"]):
        alias = f["bundle_id"].split("__")[0]
        parts.append(f"{alias}/{f['query_profile_id']}#{f['rank']} ({f['score']:.3f})")
    return " · ".join(parts) + f"  [{len(found_by)} sistemas]"


def _paragraph_lines(parent: dict, max_chars: int) -> list[str]:
    lines: list[str] = []
    for p in parent.get("paragraphs") or []:
        text = " ".join((p.get("text") or "").split())
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"  [{p.get('order')}] {text}")
    if not lines:  # parent sin párrafos (p. ej. «(Sin contenido)»)
        text = " ".join((parent.get("text") or "").split())
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"  (sin párrafos) {text}" if text else "  (Sin contenido)")
    return lines


def render_worksheet(
    pool: dict[str, dict],
    parents_by_id: dict[str, dict],
    questions_by_id: dict[str, dict],
    *,
    worksheet_top: int = 10,
    max_chars: int = 0,
) -> str:
    """Markdown por pregunta para anotar: candidatos con cita, `found_by` y párrafos numerados.

    Muestra los `worksheet_top` mejores candidatos + cualquier parent ya juzgado que quede fuera de
    ese corte (para que un juzgado nunca desaparezca). `max_chars`=0 ⇒ párrafos sin recortar.
    """
    out: list[str] = ["# Worksheet de anotación — gold de relevancia\n", _PROTOCOL_HEADER]
    for qid in sorted(pool):
        q = questions_by_id.get(qid, {})
        entry = pool[qid]
        cands = entry["candidates"]
        style, scope = q.get("query_style", "?"), q.get("answer_scope", "?")
        out.append(f"\n---\n\n## {qid} · {style} · {scope}")
        out.append(f"**Pregunta:** {q.get('query', '')}")
        out.append(
            f"_split={q.get('split', '?')} · difficulty={q.get('difficulty', '?')} · "
            f"failure_mode={q.get('failure_mode') or '—'}_"
        )
        if entry["judged_not_pooled"]:
            out.append(
                "> ⚠️ Parents ya juzgados que NINGÚN sistema recuperó (revisar recall o gold): "
                f"{', '.join(entry['judged_not_pooled'])}"
            )
        shown = cands[:worksheet_top]
        extra_judged = [c for c in cands[worksheet_top:] if c["current_relevance"] is not None]
        for c in shown + extra_judged:
            pid = c["parent_id"]
            parent = parents_by_id.get(pid, {})
            cit = parent.get("citation") or {}
            label = cit.get("label") or pid
            mark = (
                f"[JUZGADO rel={c['current_relevance']} {c['current_review_status']}]"
                if c["current_relevance"] is not None
                else "[sin juzgar]"
            )
            flags = []
            if parent.get("is_without_content"):
                flags.append("Sin contenido")
            if parent.get("is_annex"):
                flags.append("anexo")
            if parent.get("contains_table"):
                flags.append("tabla")
            flagstr = f" · flags: {', '.join(flags)}" if flags else ""
            out.append(f"\n### {label} · `{pid}` {mark}{flagstr}")
            if cit.get("url"):
                out.append(str(cit["url"]))
            out.append(f"_recuperado por:_ {_found_by_compact(c['found_by'])}")
            out.append("```")
            out.extend(_paragraph_lines(parent, max_chars))
            out.append("```")
    return "\n".join(out) + "\n"


def pool_to_jsonl_records(pool: dict[str, dict], questions_by_id: dict[str, dict]) -> list[dict]:
    """Aplana el pool a registros JSONL (uno por pregunta) con sus metadatos."""
    records: list[dict] = []
    for qid in sorted(pool):
        q = questions_by_id.get(qid, {})
        entry = pool[qid]
        records.append(
            {
                "query_id": qid,
                "split": q.get("split"),
                "query": q.get("query"),
                "query_style": q.get("query_style"),
                "answer_scope": q.get("answer_scope"),
                "failure_mode": q.get("failure_mode"),
                "candidates": entry["candidates"],
                "judged_not_pooled": entry["judged_not_pooled"],
            }
        )
    return records
