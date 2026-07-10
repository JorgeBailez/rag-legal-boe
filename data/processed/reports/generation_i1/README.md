# Batch de generación I1 — re-validación (2026-07-09/10)

Re-corrida de generación con el **perfil de consulta ganador `I1_LEGAL`** (coherencia con OE-03; antes
se usaba I2, equivalente pero incoherente). Config única y validada: `qwen2.5:7b`/`14b`, k=3,
`P_EXPAND_BOUNDED`, budgets 4000/16000, `num_ctx=8192`, `num_predict=1536`, temp 0, `--no-judge`.
`prompt_fingerprint=4d85b5f5f509` (v1) en las 10 corridas. Corrido en Colab-GPU.

## Inventario (10 corridas)

| run_id | modo | split | gen | k | n |
|---|---|---|---|---|---|
| gen_20260709T232244Z | rag | test | 7B | 3 | 28 |
| gen_20260709T232721Z | rag | development | 7B | 3 | 53 |
| gen_20260709T232954Z | rag | out_of_corpus | 7B | 3 | 40 (30 far + 10 near) |
| gen_20260709T233111Z | closed_book | test | 7B | 3 | 28 |
| gen_20260709T233304Z | oracle | test | 7B | 3 | 28 |
| gen_20260709T233759Z | rag | development | 7B | **5** | 53 (§L2 3 vs 5) |
| gen_20260709T234151Z | rag | test | 14B | 3 | 28 |
| gen_20260709T234545Z | rag | out_of_corpus | 14B | 3 | 40 |
| gen_20260709T234838Z | oracle | test | 14B | 3 | 28 |
| gen_20260709T235007Z | closed_book | test | 14B | 3 | 28 |

**Falta (no en este batch):** robustez `dev` de baselines/14B (7B closed_book/oracle dev, 14B rag/oracle
dev). Se corrió la primera versión del notebook (10 corridas), no la ampliada (14). Los baselines/14B
quedan por ahora solo en `test`.

## Métricas deterministas recomputadas (métrica endurecida + gold actual)

| corrida | answered | over-abst | key-fact [IC95] | citas F1 [IC95] |
|---|---|---|---|---|
| 7B rag test | 17/28 | 39,3 % | 0,50 [0,32–0,66] | 0,78 [0,60–0,91] |
| 7B rag dev | 31/53 | 41,5 % | 0,67 [0,49–0,85] | 0,90 [0,75–1,00] |
| 7B closed_book test | 18/28 | 35,7 % | 0,23 [0,07–0,42] | — |
| 7B oracle test | 24/28 | 14,3 % | 0,57 [0,41–0,74] | 0,99 [0,96–1,00] |
| 7B rag dev k=3 | 31/53 | 41,5 % | 0,67 | 0,90 |
| 7B rag dev k=5 | 36/53 | 32,1 % | 0,60 | 0,88 |
| 14B rag test | 23/28 | 17,9 % | 0,65 [0,52–0,78] | 0,77 [0,61–0,90] |
| 14B oracle test | 26/28 | 7,1 % | 0,68 [0,57–0,80] | 0,95 [0,90–0,99] |
| 14B closed_book test | 1/28 | 96,4 % | — | — |

## Seguridad: false-answer sobre OOC (far vs near-miss)

| | far-domain (30) | near-miss (10) |
|---|---|---|
| 7B | 0/30 | 1/10 (q92nm_001) |
| 14B | 3/30 (q92o_002/007/018) | 4/10 (q92nm_001/004/007/010) |

**Hallazgo clave (lo desbloquea el near-miss):** el "0 respuestas indebidas" del 7B era sobre
far-domain; en near-miss se le escapa 1/10. El 14B, 3/30 en far-domain, se dispara a **4/10 en
near-miss** → la regresión de seguridad del modelo grande es mucho peor en el caso realista.

## 7B vs 14B — el 14B es MEJOR generador (aunque pierda en OOC)

El 14B no solo responde más: es mejor generador **en las mismas preguntas** y más disciplinado.

- **Key-fact PAREADO (mismas preguntas respondidas por ambos, con gold):**
  - `rag` test (n=17): 7B **0,50** → 14B **0,63**, Δ **+0,13** [+0,01, +0,26], **p=0,026 (SIGNIFICATIVO)**.
  - `oracle` test (n=23): 7B 0,59 → 14B 0,66, Δ +0,07 [−0,03, +0,18] (n.s.; con evidencia perfecta el hueco se estrecha).
- **Responde más (menos sobre-abstención):** 17,9 % (14B) vs 39,3 % (7B) → 23/28 vs 17/28 respondidas.
- **No inventa de memoria (disciplina closed-book):** sin evidencia, el 14B se abstiene en 27/28
  (responde 1); el **7B responde 18/28 confabulando de memoria** (key-fact 0,23 = inventa). Este es un
  punto fuerte del 14B poco visible en las medias.
- **Citas ≈ iguales:** F1 0,77 (14B) vs 0,78 (7B).

**Único punto donde pierde: seguridad OOC.** El 14B sobre-confía en la evidencia recuperada-pero-
insuficiente → responde 3/30 far y **4/10 near-miss** (vs 7B 0/30 y 1/10). Ojo al matiz coherente: el
14B NO inventa de memoria (closed-book 1/28) pero SÍ se fía de más de un casi-acierto recuperado. →
La lectura correcta para §gen-tamaño no es "14B descalificado", sino **"14B es mejor generador; el
único blocker es la sobre-respuesta OOC, que ataca una señal de suficiencia"** (línea ya apuntada en
conclusiones: modelo más capaz + gate de suficiencia).

## Coherencia con los números viejos (I2): LA HISTORIA AGUANTA

Casi idénticos: 7B key-fact 0,50 (=), citas 0,78 (≈0,79), over-abst 35,7→39,3 % (18→17 answered);
baselines closed-book 0,21→0,23, oracle 0,57/0,99 (=); 14B over-abst 17,9 % (=), far false-answer
3/30 (=). Conclusiones intactas y ahora **coherentes con I1 (OE-03)**.

## Caveats (importantes)

1. **Reproducibilidad de la generación.** De 53 dev, 33 idénticas a la corrida I2; de las 20 que
   cambian, **10 por recuperación distinta (efecto real de I1) y 10 con la MISMA evidencia** (ruido de
   Ollama entre sesiones aun a temp 0). Material honesto de reproducibilidad para la memoria.
2. **Anotación L3/L5 (§gen-juez).** Está atada a las respuestas I2. Bajo I1 cambian 17/30 anotadas
   (~8 con evidencia distinta = juicio nuevo; ~9 mismo-evidencia/paráfrasis = confirmación rápida).
   Lista en `PROGRESO.md`. Decidir: re-anotar esas 17 (coherente) vs mantener §gen-juez sobre I2 con nota.
3. **§L2 (3 vs 5) se suaviza.** Con métrica endurecida + top-k directo: key-fact 0,67 (k3) vs 0,60 (k5)
   —antes 0,74→0,57—. k=3 sigue mejor en key-fact/citas, pero k=5 responde más (over-abst 32 vs 41 %);
   el relato "más contexto claramente peor / lost-in-the-middle" hay que atemperarlo.
