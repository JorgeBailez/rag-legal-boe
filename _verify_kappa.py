"""Verificación autónoma (stdlib) del nuevo cálculo de κ sobre los 32 pares reales."""

import json
import random
from collections.abc import Sequence


def confusion(human, judge, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    m = [[0] * k for _ in range(k)]
    for h, j in zip(human, judge, strict=True):
        m[idx[h]][idx[j]] += 1
    return m


def kappa(m, *, mode):
    k = len(m)
    n = sum(sum(r) for r in m)
    if n == 0:
        return None
    rows = [sum(m[i]) for i in range(k)]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]
    denom = (k - 1) or 1

    def w(i, j):
        if i == j:
            return 0.0
        if mode == "nominal":
            return 1.0
        if mode == "linear":
            return abs(i - j) / denom
        return (i - j) ** 2 / (denom ** 2)  # quadratic

    obs = sum(w(i, j) * m[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 if exp == 0 else 1.0 - obs / exp


def kappa_ci(pairs, labels, *, mode, n_boot=2000, seed=42, alpha=0.05):
    n = len(pairs)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        k = kappa(confusion([h for h, _ in s], [j for _, j in s], labels), mode=mode)
        if k is not None:
            stats.append(k)
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return lo, hi


rows = [json.loads(line) for line in open("anotacion_juez.jsonl", encoding="utf-8")]
order = ("incorrect", "partial", "correct")

corr = [
    (r["human_correctness"], r["judge_correctness"])
    for r in rows
    if r.get("human_correctness") in order and r.get("judge_correctness") in order
]
human = [h for h, _ in corr]
judge = [j for _, j in corr]
m = confusion(human, judge, order)
n = len(corr)
agree = sum(1 for h, j in corr if h == j)

print(f"CORRECCIÓN — n={n} · acuerdo={agree}/{n}={agree/n:.1%}")
print(f"  orden ordinal: {order}")
print("  matriz de confusión (filas=humano, columnas=juez):")
for i, lab in enumerate(order):
    print(f"    {lab:>9} | {m[i]}")
print(f"  margen humano  : { {order[i]: sum(m[i]) for i in range(3)} }")
print(f"  margen juez    : { {order[j]: sum(m[i][j] for i in range(3)) for j in range(3)} }")
for mode in ("nominal", "linear", "quadratic"):
    k = kappa(m, mode=mode)
    lo, hi = kappa_ci(corr, order, mode=mode)
    print(f"  κ {mode:<9} = {k:.3f}   IC95% bootstrap=[{lo:.2f}, {hi:.2f}]")

# Faithfulness (binaria) — solo los que tienen human_faithful anotado
fl = ("unfaithful", "faithful")
faith = [
    ("faithful" if r["human_faithful"] else "unfaithful",
     "faithful" if r["judge_faithful"] else "unfaithful")
    for r in rows
    if isinstance(r.get("human_faithful"), bool) and isinstance(r.get("judge_faithful"), bool)
]
if faith:
    fm = confusion([h for h, _ in faith], [j for _, j in faith], fl)
    fa = sum(1 for h, j in faith if h == j)
    print(f"\nFIDELIDAD — n={len(faith)} · acuerdo={fa}/{len(faith)}={fa/len(faith):.0%}")
    print(f"  κ nominal = {kappa(fm, mode='nominal'):.3f}  (binaria → ponderada = nominal)")
    print(f"  margen humano: { {fl[i]: sum(fm[i]) for i in range(2)} }  margen juez: "
          f"{ {fl[j]: sum(fm[i][j] for i in range(2)) for j in range(2)} }")

# WHAT-IF: si al completar las referencias los 9 desacuerdos correct/partial pasan a correct/correct
print("\nWHAT-IF (ilustrativo) — referencias completas resuelven los 9 'humano=correct/juez=partial':")
wif = []
flipped = 0
for h, j in corr:
    if h == "correct" and j == "partial" and flipped < 9:
        wif.append(("correct", "correct"))
        flipped += 1
    else:
        wif.append((h, j))
wm = confusion([h for h, _ in wif], [j for _, j in wif], order)
wa = sum(1 for h, j in wif if h == j)
print(f"  acuerdo={wa}/{n}={wa/n:.0%} · κ nominal={kappa(wm, mode='nominal'):.3f} · "
      f"κ lineal={kappa(wm, mode='linear'):.3f}")
