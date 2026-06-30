# Análisis de errores de la generación

La revisión que sigue mira, una a una, las respuestas que da el sistema sobre el conjunto de desarrollo del banco de pruebas (53 preguntas). No busca una nota de calidad —eso lo dan las métricas—, sino entender dónde se equivoca y por qué, que es lo que indica qué merece la pena arreglar.

Conviene situar el experimento. El sistema admite dos formas de pedirle prudencia al modelo: una versión conservadora, que se abstiene en cuanto duda, y una permisiva, que responde siempre que la evidencia recuperada dé pie a ello. La conservadora es la que queda como operativa, porque la permisiva, aunque resulta más útil, rompe una garantía que no se puede ceder: no inventar respuestas cuando la pregunta cae fuera del corpus. El análisis se hace justo sobre la versión permisiva, y a propósito: al obligarla a mojarse, saca a la luz los errores que la conservadora tapa callándose.

Sobre las 53 preguntas, la versión permisiva respondió 48 y se abstuvo en 5. De esas 48, veintiséis tenían respuesta de referencia en el gold y se contrastaron con ella; las otras veintidós se comprobaron contra el texto legal vigente y la coherencia interna de la respuesta.

## Panorama general

Unas 38 respuestas (en torno al 80 %) son correctas y están fundamentadas: las cifras y los artículos coinciden con la norma y citan el bloque legal esperado. Las diez restantes presentan algún problema, y de ellas cuatro son invenciones de dato en sentido estricto, es decir, el modelo afirma algo que la norma no dice.

Lo importante es que la ganancia de utilidad es real. Al pasar de la versión conservadora a la permisiva, la abstención indebida —callarse cuando sí había con qué responder— cae del 46 % al 9 %, y ese margen recuperado es en su mayoría respuesta buena, no humo. El precio es una cola de errores y, sobre todo, que el sistema pasa a responder al 30 % de las preguntas fuera de corpus en lugar de abstenerse. Por eso la versión permisiva no es adoptable tal cual.

## Los dos patrones de error

Casi todos los fallos caen en uno de dos moldes.

El primero es responder desde el artículo equivocado por un choque en la recuperación. El buscador trae un artículo parecido pero de otra ley o de otra materia, y el prompt permisivo responde desde él con aplomo. Es el mismo problema de colisión de número de artículo que aparece en la evaluación de recuperación, aquí ya convertido en una respuesta errónea. Le ocurre, por ejemplo, a una pregunta sobre recargos tributarios que acaba contestada con un artículo del impuesto de sociedades.

El segundo es la sobre-extensión del modelo: la recuperación es correcta, pero el modelo añade un detalle plausible que la evidencia no respalda. Es la tendencia de un modelo pequeño a "rellenar" para sonar completo, como inventar un requisito de antigüedad en la pensión de viudedad.

La distinción importa para decidir cómo atajarlos. El primer patrón lo frena, al menos en parte, un umbral sobre la confianza de la recuperación: cuando el buscador no está seguro, el sistema se abstiene. El segundo no lo detecta ningún filtro de recuperación, porque ahí lo recuperado es correcto y el desliz lo pone el modelo al redactar; solo lo corta un prompt más estricto o una comprobación explícita de que cada afirmación se apoya en la evidencia.

## Casos concretos

| Pregunta | Qué falla |
|---|---|
| q92_017 — recargos tributarios | Responde con un artículo de otra ley (impuesto de sociedades): mismo número, materia distinta. La versión conservadora se abstenía. |
| q92_066 — base de cotización | Confunde la base de cotización con la fórmula de la base reguladora de la pensión. |
| q92_063 / q92_064 — viudedad de pareja de hecho | Añade un requisito de antigüedad que la norma no exige. |
| q92_060 — intimidad y secretos | Incluye un artículo sobre secretos de la Defensa Nacional, ajeno a la pregunta. |
| q92_004 — vacaciones | Da el dato correcto (30 días) pero le suma un régimen de funcionarios que no corresponde. |
| q92_036 — protección de datos | Cita una ley ya derogada y, además, deja escritos en la respuesta los identificadores internos de la evidencia. |

## Dos matices de medición

La métrica automática de hechos clave compara texto casi literal, así que penaliza la paráfrasis y los cambios de formato: "a instancia tuya" frente a "de la persona interesada", o "19 %" frente a "19 por ciento", cuentan como fallo aunque la respuesta sea correcta. Por eso esa cifra (0,76 aquí) se lee mejor como un suelo pesimista que como la corrección real.

Aparte, en una respuesta el modelo dejó escritos los marcadores internos "(E1)/(E2)" que el sistema usa para señalar las evidencias. Es un fallo menor de seguimiento de instrucciones, no de contenido.

## Hasta dónde llega esta revisión

Las 26 preguntas con gold se validaron contra la referencia; las 22 sin gold, contra el texto legal y la coherencia interna, no contra la evidencia exacta que vio el modelo, que no se guardó en esta corrida. Las invenciones señaladas convendría confirmarlas reconstruyendo esa evidencia antes de darlas por seguras.
