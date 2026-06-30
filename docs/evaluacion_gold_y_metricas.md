# Diseño de la evaluación: dataset de referencia y métricas

Este documento define cómo se evalúa el sistema: qué se mide, con qué dataset y cómo se construye ese dataset, de modo que las comparaciones entre estrategias de recuperación o entre configuraciones del generador se apoyen en evidencia y no en impresiones.

El principio de fondo viene de un hallazgo de la literatura sobre herramientas legales con inteligencia artificial: el estudio del RegLab de Stanford (Magesh et al., 2024-2025) midió entre un 17 % y un 33 % de respuestas con invenciones en herramientas comerciales basadas en recuperación, y un 43 % en un modelo general. La lección para un sistema legal es que una respuesta puede ser plausible, estar bien citada en la forma y aun así ser infiel a la fuente o directamente falsa. Por eso la evaluación separa de forma explícita tres cosas que no son lo mismo: que la respuesta sea correcta, que esté fundada en la evidencia entregada y que esté bien citada.

## La evaluación por capas

El sistema se evalúa por capas desacopladas, de la recuperación a la respuesta final. La razón es poder localizar el fallo: si una respuesta es mala, conviene saber si el sistema recuperó mal, si ensambló mal el contexto, si el modelo inventó, si citó mal o si se abstuvo cuando no debía. Las seis capas son las siguientes.

**Capa 1, recuperación.** Mide si el bloque legal correcto aparece y aparece arriba en la lista de resultados. La métrica primaria es el nDCG@10, que aprovecha la relevancia graduada del gold (un bloque puede ser central, de apoyo o irrelevante). Como controles se usan la cobertura (recall) a distintas profundidades, el acierto en la primera posición y el rango recíproco medio. A nivel más fino, dos métricas miden cuántos de los párrafos concretos que sustentan la respuesta se recuperan, lo que exige anotar en el gold los párrafos reales de evidencia.

**Capa 2, calidad del contexto.** Mide si el contexto que se ensambla y se pasa al generador contiene la evidencia necesaria y poco ruido. Se mide la cobertura de la evidencia, la proporción de fragmentos del contexto que son realmente relevantes y la redundancia. Esto se puede medir sin necesidad de un juez, usando el gold de relevancia.

**Capa 3, fidelidad.** Es la métrica número uno en el ámbito legal. Mide si cada afirmación de la respuesta se deriva de la evidencia entregada al modelo. La definición operativa, en la línea de los marcos RAGAS y ALCE, consiste en descomponer la respuesta en afirmaciones atómicas y medir qué fracción está respaldada por la evidencia:

```
fidelidad = afirmaciones respaldadas por el contexto / total de afirmaciones
```

No necesita una respuesta de referencia, solo la respuesta y el contexto. Es la métrica que captura la invención legal: el modelo se inventa un plazo, un umbral o una excepción que no estaba en la evidencia. Por diseño, el objetivo del sistema es una fidelidad cercana a uno, porque es un sistema a prueba de fallos.

**Capa 4, calidad de la cita.** El sistema ya garantiza por contrato que los identificadores citados pertenecen a las evidencias entregadas, pero eso es solo correcto en la forma. Lo que importa en derecho es si el bloque citado respalda de verdad la afirmación. Se mide si toda afirmación que necesita respaldo tiene al menos una cita que la sostiene (cobertura de cita), si cada cita aportada es realmente necesaria (precisión de cita) y si el conjunto de bloques citados coincide con el esperado en el gold. El verificador, sea un modelo de inferencia textual o un modelo de lenguaje como juez, debe validarse contra una muestra anotada a mano.

**Capa 5, corrección.** Mide si la respuesta dice los hechos correctos. Hay dos vías. La primera es barata y robusta: el gold lista los hechos clave que una respuesta correcta debe contener (por ejemplo, "un mes", "40.000 euros", "cuatro años") y se mide qué fracción aparece, sin comparar la frase literal. La segunda es una comparación semántica entre la respuesta y la respuesta de referencia, hecha por un juez y validada contra anotación humana. Opcionalmente, el gold puede incluir hechos prohibidos para detectar premisas falsas.

**Capa 6, abstención.** Mide si el sistema se calla cuando debe, y solo cuando debe. Tiene dos tipos de error que no son simétricos. Ante una pregunta fuera del corpus, lo que importa es la tasa de abstención (objetivo: cercana al cien por cien), porque responder ahí es el error peligroso en derecho. Ante una pregunta respondible, lo que importa es no abstenerse de más; ese error es molesto pero seguro. Se reportan por separado y como exactitud equilibrada, dando el mismo peso a responder y a abstenerse. Conviene además distinguir los dos puntos en que el sistema puede abstenerse: el determinista, antes de llamar al modelo cuando no hay evidencia, y el decidido por el propio modelo. Con el gold se puede trazar la curva de exactitud frente a cobertura y calibrar un umbral mínimo de confianza de recuperación.

A estas seis capas se añade una transversal de coste: latencia, número de tokens y memoria.

## El dataset de referencia

El dataset se organiza en tres ficheros, separando el gold de recuperación del de generación.

**Las preguntas** (`questions.jsonl`) llevan, además del texto, su tipo (por número de artículo, ciudadana, conceptual, procedimental, léxica, comparativa o sin respuesta), su dificultad, el modo de fallo que pretenden provocar y su procedencia. Un campo de estado de revisión indica si una pregunta es un borrador o ya está revisada por una persona, y debe reflejar la realidad: una pregunta generada automáticamente es un borrador hasta que alguien la revisa.

**Los juicios de relevancia** (`judgments.jsonl`) usan tres grados: central o suficiente, de apoyo o matiz, y revisado y descartado (un bloque que el recuperador confunde pero que no responde, es decir, un negativo difícil). Cada juicio relevante anota los párrafos reales que lo sustentan y una cita literal. Una pregunta cuya respuesta requiere varios bloques se marca como tal y necesita al menos dos bloques relevantes.

**Las claves de respuesta** (`answer_keys.jsonl`) son el gold de generación, independiente del modelo. Por cada pregunta recogen si es respondible, una respuesta de referencia breve, los hechos clave que debe contener, los hechos prohibidos que delatarían una invención y los bloques que sería correcto citar. Para las preguntas fuera de corpus, la clave marca que no son respondibles y sirve de gold de abstención. Un ejemplo:

```json
{
  "query_id": "q0016",
  "answerable": true,
  "reference_answer": "El plazo para interponer el recurso de alzada es de un mes si el acto es expreso.",
  "key_facts": ["un mes", "acto expreso"],
  "forbidden_facts": ["tres meses", "quince días"],
  "expected_citation_parents": ["BOE-A-2015-10565__a122"],
  "answer_scope": "single_parent",
  "review_status": "reviewed",
  "notes": "Artículo 122.1 de la Ley 39/2015."
}
```

## Cómo se escriben buenas preguntas

El corpus es un conjunto temático interrelacionado, lo que permite preguntas cruzadas y trampas realistas. La batería se diseña como una matriz de tipo de pregunta por norma por modo de fallo, estratificada por dificultad. Cada modo de fallo existe para forzar una parte concreta del sistema.

1. **Directa por artículo**, para la recuperación exacta: "¿Qué establece el artículo 122 de la Ley 39/2015 sobre el recurso de alzada?".
2. **Ciudadana o en paráfrasis**, para la distancia entre el lenguaje del ciudadano y el jurídico, que es el terreno de la recuperación densa: "Si pido algo a la Administración y no me contestan, ¿es un sí o un no?".
3. **Numérica o de umbral**, para la extracción de una cifra y la fidelidad de no inventar números: "¿A partir de qué importe un contrato de obras deja de ser menor?".
4. **Conceptual o de definición**: "¿Qué es una sede electrónica?".
5. **Procedimental o de varios pasos**: "¿Qué pasos sigue la aprobación de una ordenanza local?".
6. **Léxica o de término técnico**, que favorece a la recuperación léxica y marca dónde flojea la densa: "¿Qué es la Base de Datos Nacional de Subvenciones?".
7. **Comparativa entre normas**, que exige varios bloques y desambiguar: "¿En qué se diferencian los plazos del recurso de alzada y del de reposición?".
8. **Condicional o de varios saltos**: "Una empresa va a licitar una obra de 600.000 euros, ¿necesita clasificación?".
9. **De negación o excepción**: "¿Quién no puede contratar con el sector público?".
10. **De desambiguación**, una pregunta infra-especificada que el sistema no debería resolver al azar: "¿Cuál es el plazo del recurso?", sin decir cuál.
11. **De premisa falsa**, para comprobar si el sistema corrige o asiente: "¿Por qué el plazo del recurso de alzada es de tres meses?", cuando en realidad es de un mes.
12. **De trampa temporal**, sobre un artículo vaciado por una reforma: el sistema no debe servir la redacción histórica.
13. **Dentro del corpus pero sin respuesta**, una abstención fina distinta de la de fuera de corpus: un tema que trata una norma pero cuyo detalle concreto no aparece.
14. **Fuera de corpus**, la abstención gruesa: "Horario de la oficina de extranjería de Cuenca".
15. **De robustez ante la paráfrasis**: varias redacciones del mismo asunto, para medir la varianza.

Como reglas de calidad, las variantes de un mismo asunto se agrupan bajo un identificador común y no se reparten entre las particiones de desarrollo y de prueba, para evitar que el sistema "vea" en desarrollo algo que se le va a preguntar en prueba. Conviene cubrir cada modo de fallo en varias normas, no solo en una. Y la partición de prueba es sagrada: las afirmaciones finales del trabajo se hacen sobre ella, y no se mira durante el desarrollo.

## Cómo se construye el gold

El dataset se construye con un proceso reproducible y con una persona en el bucle, evitando el sesgo de evaluar el sistema con lo que el propio sistema ya encuentra.

1. **Redacción de candidatas.** Se generan preguntas asistidas por un modelo, por combinación de norma, artículo, tipo y modo de fallo. Nacen como borrador.
2. **Agrupación de candidatos a juzgar.** Para cada pregunta, los bloques que se van a juzgar se reúnen desde varios sistemas (la recuperación densa con distintas vistas y la léxica), no solo desde el recuperador actual. Es el método de agrupación de la evaluación de recuperación: evita que el gold favorezca al recuperador que ya se tiene. Sin este paso, una comparación entre recuperadores estaría sesgada de origen.
3. **Anotación humana.** Para cada par de pregunta y bloque se asigna la relevancia, se marcan los párrafos reales y la cita. Para cada pregunta se escribe la respuesta de referencia, los hechos clave, los bloques a citar y si es respondible. Solo entonces la entrada pasa a estar revisada.
4. **Acuerdo entre anotadores.** Aunque el anotador principal sea una sola persona, una segunda pasada sobre una muestra y el cálculo de un coeficiente de acuerdo dan credibilidad. Como mínimo, debe existir un protocolo documentado para resolver los casos dudosos.
5. **Validación automática.** Un validador comprueba el contrato de los tres ficheros y unos mínimos de cobertura por partición antes de dar por buena una comparación.

### Cómo usar un modelo como juez sin engañarse

Para medir fidelidad, citas y corrección a escala se recurre a un modelo de lenguaje como juez. Para que sus números valgan, hay salvaguardas que no son opcionales:

- **Validarlo contra anotación humana** en una muestra y reportar el acuerdo. La literatura muestra que las métricas automáticas de cita correlacionan con el juicio humano cuando se validan, no por defecto.
- **El juez no puede ser el generador**, ni de su misma familia, por el sesgo de auto-preferencia. Conviene un modelo distinto y, a ser posible, más capaz.
- **Determinismo**: temperatura cero, prompt versionado y salida estructurada.
- **Trazabilidad**: registrar siempre qué juez y qué versión se usaron.

En este proyecto, esta validación se llevó a cabo y arrojó un resultado importante: el juez disponible resultó insuficiente para fidelidad y corrección, así que esas dos dimensiones se reportan a partir de anotación humana. El detalle está en el registro de decisiones de diseño.

## Fuentes

- RAGAS, métricas de fidelidad, relevancia y precisión y cobertura del contexto:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- ALCE, generación de texto con citas y su evaluación por inferencia textual:
  <https://arxiv.org/abs/2305.14627>
- LegalBench-RAG, banco de recuperación en el ámbito legal:
  <https://arxiv.org/abs/2408.10343>
- Magesh et al., sobre la fiabilidad de las herramientas legales de inteligencia artificial (RegLab, Stanford):
  <https://arxiv.org/abs/2405.20362>
- Trabajos sobre abstención y preguntas sin respuesta en sistemas de recuperación:
  <https://arxiv.org/pdf/2412.12300>
