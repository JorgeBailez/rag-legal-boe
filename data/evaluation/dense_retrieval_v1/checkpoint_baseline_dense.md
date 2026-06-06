# Checkpoint experimental — baseline denso

Registro reproducible del checkpoint de selección del recuperador denso sobre el
corpus MVP de 10 normas.

## Alcance

- Evaluación de checkpoint, no evaluación formal definitiva.
- Split utilizado para la selección: `development`.
- Métrica primaria: `ParentnDCG@10`.
- Recuperación dense-only con índice exacto.
- Sin ensamblado de contexto y sin generación con LLM.

## Selección provisional de modelo y perfil de consulta

Baseline provisional:

```text
bundle: e5-large-instruct__j1__bc11142bdcc5
modelo: e5-large-instruct
vista documental: J1
perfil de consulta: I2_CITIZEN_LEGISLATION
ParentnDCG@10: 0.8694
```

Alternativa de sensibilidad:

```text
modelo: e5-large-instruct
perfil de consulta: I1_LEGAL
ParentnDCG@10: 0.8662
```

Bootstrap pareado entre los dos perfiles:

```text
diferencia media I2 - I1: 0.0033
IC95: [-0.0336, 0.0369]
```

El intervalo cruza cero. No existe separación concluyente entre ambos perfiles.
Se conserva `I2_CITIZEN_LEGISLATION` como baseline provisional por su alineación
con las consultas ciudadanas y `I1_LEGAL` como alternativa de sensibilidad.

## Ablación de vistas documentales

Comparación controlada manteniendo constantes:

```text
modelo: e5-large
perfil de consulta: BASELINE
split: development
```

| Vista | Representación | Vectores | ParentnDCG@10 | ParentRecall@5 | EvidenceRecall@5 | ParentHit@1 | DuplicateParentRate@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| J1 | Chunk enriquecido con contexto jurídico | 3300 | 0.8451 | 0.9750 | 0.8750 | 0.6750 | 0.1800 |
| J2 | Texto crudo del child | 3300 | 0.8304 | 0.9250 | 0.8750 | 0.7000 | 0.1450 |
| C1 | Ventanas token-aware derivadas del parent | 2627 | 0.7294 | 0.8750 | 0.8250 | 0.5500 | 0.0750 |

Bootstrap pareado:

| Comparación | Diferencia media | IC95 | Lectura |
|---|---:|---|---|
| J1 - J2 | 0.0146 | [-0.0361, 0.0695] | Sin separación concluyente |
| J1 - C1 | 0.1157 | [0.0261, 0.2215] | J1 superior |
| J2 - C1 | 0.1011 | [0.0117, 0.1996] | J2 superior |

## Decisión

- Seleccionar `J1` como vista documental del baseline denso.
- Conservar `J2` y `C1` como ablaciones documentadas.
- No extender `J2` ni `C1` a otros modelos.
- Mantener como baseline provisional:
  `e5-large-instruct · J1 · I2_CITIZEN_LEGISLATION`.
- Conservar `I1_LEGAL` como alternativa de sensibilidad.
- Ampliar Gate C antes de una evaluación formal definitiva.
- Posponer ensamblado de contexto y generación con LLM hasta cerrar el bloque
  de recuperación densa.
