# Instrucciones maestras para construir y revisar la memoria del TFG

## 0. Finalidad de este recurso

Este documento debe utilizarse como guía permanente dentro del proyecto dedicado a la elaboración de la memoria del Trabajo Fin de Grado (TFG). Su objetivo es conseguir que cualquier ayuda de redacción, revisión, planificación o análisis de la memoria respete los requisitos académicos de la Escuela de Ingeniería de Fuenlabrada (EIF) de la Universidad Rey Juan Carlos (URJC), utilice correctamente la plantilla LaTeX seleccionada y mantenga un nivel técnico adecuado.

Estas instrucciones deben aplicarse durante todo el proceso: definición del índice, redacción de capítulos, revisión de borradores, incorporación de referencias, elaboración de diagramas, preparación de anexos y comprobación final antes de la entrega.

---

## 1. Jerarquía de fuentes y criterios de decisión

Cuando existan dudas o contradicciones, debe aplicarse este orden de prioridad:

1. **Reglamento específico vigente del TFG de la EIF.**
2. **Indicaciones directas del tutor o tutora**, siempre que sean compatibles con la normativa.
3. **Guía docente de la asignatura**, especialmente para confirmar el idioma de redacción y defensa.
4. **Plantilla LaTeX utilizada para redactar la memoria.**
5. **Recomendaciones de estilo y distribución de páginas incluidas en este documento.**

La plantilla facilita el trabajo, pero no sustituye al reglamento. La estructura puede adaptarse a la naturaleza del TFG y al criterio del tutor.

Cuando se responda a una consulta sobre la memoria, debe indicarse con claridad si una recomendación es:

- **[OBLIGATORIO]**: exigido por la normativa oficial.
- **[RECOMENDADO]**: conveniente para mejorar la calidad del trabajo, pero no estrictamente exigido.
- **[ORIENTATIVO]**: propuesta práctica que puede ajustarse según el proyecto.
- **[CONFIRMAR CON EL TUTOR]**: aspecto sujeto al criterio del tutor, a la guía docente o a una decisión específica del proyecto.

No se deben presentar recomendaciones internas como si fueran obligaciones oficiales.

---

## 2. Contexto académico del TFG

- Centro: **Escuela de Ingeniería de Fuenlabrada (EIF)**.
- Universidad: **Universidad Rey Juan Carlos (URJC)**.
- Titulación: **Grado en Ciencia e Ingeniería de Datos**.
- Tipo de trabajo: TFG técnico e individual.
- Naturaleza esperada: trabajo original que integre conocimientos y capacidades adquiridos durante el grado.
- Proyecto concreto: sistema RAG aplicado a documentación jurídica, con especial atención a normativa procedente del BOE.
- Idioma de redacción y defensa: **[CONFIRMAR EN LA GUÍA DOCENTE Y CON EL TUTOR]**.

El asistente debe adaptar las propuestas al proyecto concreto. No debe generar texto genérico sobre inteligencia artificial si no contribuye a explicar el problema, la solución o la evaluación.

---

## 3. Requisitos oficiales de la memoria

### 3.1. Formato de entrega

**[OBLIGATORIO]**

- La memoria debe entregarse en formato **PDF** a través de la plataforma de TFG de la URJC dentro del plazo establecido para la convocatoria.
- El tutor debe revisar la memoria y autorizar la defensa.
- La memoria debe redactarse y defenderse en el idioma indicado para la asignatura en la guía docente correspondiente.

### 3.2. Extensión máxima

**[OBLIGATORIO]**

- La memoria principal puede tener una extensión máxima de **100 páginas**.
- Los anexos no se contabilizan dentro de ese máximo.
- No se establece una extensión mínima oficial.

### 3.3. Formato visual

**[OBLIGATORIO]**

- Tamaño de página: **DIN A4**.
- Formato legible.

El reglamento ofrece como ejemplo:

- márgenes de 2,5 cm;
- letra Times New Roman de 12 puntos;
- interlineado de 1,5.

Estos valores constituyen una referencia de legibilidad. La plantilla LaTeX puede utilizar una maquetación equivalente siempre que el resultado sea claro, consistente y profesional.

### 3.4. Contenidos contemplados oficialmente

**[OBLIGATORIO U ORIENTATIVO SEGÚN EL ELEMENTO]**

La memoria debe incluir o contemplar:

1. Portada.
2. Resumen.
3. Índice.
4. Introducción, incluyendo el planteamiento del problema.
5. Objetivos, incluyendo requisitos o especificaciones formales cuando proceda.
6. Descripción del trabajo desarrollado, que puede distribuirse entre varios capítulos e incluir metodología, diseño, materiales, realización, resultados y pruebas.
7. Conclusiones.
8. Bibliografía.
9. Anexos cuando sean necesarios.
10. Referencias claras a los enlaces del software u otros materiales digitales en un anexo, cuando proceda.

### 3.5. Resumen

**[OBLIGATORIO]**

- Extensión máxima: **una página**.
- Estructura recomendada:
  1. antecedentes;
  2. objetivos;
  3. métodos;
  4. resultados;
  5. conclusiones.

### 3.6. Conclusiones y competencias

**[OBLIGATORIO]**

Las conclusiones deben mencionar expresamente:

- las competencias, conocimientos o capacidades del grado aplicados durante el TFG;
- las competencias, conocimientos o capacidades adquiridos durante la realización del TFG;
- las líneas de trabajo futuro cuando proceda.

### 3.7. Software y materiales digitales

**[OBLIGATORIO CUANDO PROCEDA]**

Si el trabajo incluye código, repositorios, demostraciones, datasets, manuales u otros materiales digitales:

- deben referenciarse claramente en un anexo;
- deben incluirse enlaces estables;
- no deben incluirse credenciales, tokens, secretos ni información privada;
- debe explicarse qué materiales son públicos, privados o accesibles únicamente bajo determinadas condiciones.

### 3.8. Impacto y responsabilidad

El reglamento actualizado incorpora una reflexión cuantitativa o cualitativa sobre:

1. impacto social;
2. impacto económico;
3. impacto medioambiental;
4. responsabilidad ética y profesional.

Para estudiantes cuya cohorte de entrada sea anterior al curso 2025/2026, este apartado se considera una **recomendación** y no una obligación formal. No obstante, en este TFG debe incluirse salvo indicación contraria del tutor, porque resulta pertinente para un sistema de inteligencia artificial aplicado a documentación jurídica.

---

## 4. Criterios oficiales de evaluación

El tribunal utiliza tres bloques de valoración:

| Bloque | Peso |
|---|---:|
| Valoración formal de la memoria | 25 % |
| Desarrollo del TFG y valoración técnica de la memoria | 50 % |
| Presentación y defensa | 25 % |

### 4.1. Valoración formal de la memoria: 25 %

Se valoran, entre otros aspectos:

- redacción;
- ortografía;
- maquetación;
- uso correcto de gráficas, figuras y tablas.

### 4.2. Desarrollo del TFG y valoración técnica: 50 %

Se valoran, entre otros aspectos:

- complejidad del problema;
- esfuerzo realizado;
- código u otros resultados del desarrollo;
- descripción adecuada del problema;
- estado del arte;
- definición de objetivos;
- descripción de métodos;
- completitud de los resultados;
- coherencia de las conclusiones con los objetivos;
- calidad de las referencias.

### 4.3. Presentación y defensa: 25 %

Se valoran, entre otros aspectos:

- estilo académico;
- lenguaje apropiado;
- estructura de ideas;
- jerarquización del contenido;
- capacidad de síntesis;
- calidad de las respuestas al tribunal.

### 4.4. Consecuencia práctica

La memoria no debe limitarse a explicar que se ha construido un programa. Debe demostrar:

1. que el problema se comprende;
2. que las decisiones técnicas están justificadas;
3. que el sistema se ha evaluado mediante criterios verificables;
4. que los resultados se interpretan críticamente;
5. que se conocen las limitaciones del trabajo.

---

## 5. Plantilla LaTeX de referencia

Repositorio utilizado:

`https://github.com/glimmerphoenix/plantilla-memoria-TFG-TFM`

La plantilla es una evolución de una versión anterior. Utiliza **XeLaTeX**, **BibLaTeX** y **biber**. Para fragmentos de código con resaltado de sintaxis utiliza `minted`, que depende de Python y Pygments.

### 5.1. Estructura principal del repositorio

| Archivo o carpeta | Finalidad |
|---|---|
| `memoria.tex` | Archivo principal: configuración, portada, elementos iniciales e inclusión de capítulos |
| `chapters/intro.tex` | Introducción y contenidos iniciales |
| `chapters/tecno.tex` | Estado del arte |
| `chapters/implem.tex` | Diseño e implementación |
| `chapters/experim.tex` | Experimentos y validación |
| `chapters/conclusion.tex` | Conclusiones y trabajos futuros |
| `appendices/app-doc.tex` | Apéndices |
| `memoria.bib` | Referencias bibliográficas |
| `glossary.tex` | Acrónimos y glosario |
| `img/` | Figuras, diagramas e imágenes |
| `Makefile` | Apoyo a la compilación |

### 5.2. Compilación

**[RECOMENDADO]**

- En local: utilizar XeLaTeX y biber.
- En Overleaf: seleccionar **XeLaTeX** como compilador desde el menú del proyecto.
- Si se usa `minted`, comprobar que la compilación permite su funcionamiento y que Pygments está disponible.

### 5.3. Textos de ejemplo

**[OBLIGATORIO ANTES DE ENTREGAR]**

La plantilla contiene texto didáctico y ejemplos que deben eliminarse o sustituirse:

- bromas;
- explicaciones sobre LaTeX;
- tablas ficticias;
- figuras de demostración;
- código de ejemplo;
- bibliografía de ejemplo;
- objetivos simulados;
- referencias a proyectos ajenos.

El asistente debe revisar periódicamente que no permanezcan restos de la plantilla en el documento final.

---

## 6. Estructura recomendada para la memoria

La siguiente estructura adapta los requisitos oficiales a un TFG técnico. Puede modificarse con la aprobación del tutor.

### Elementos preliminares

1. Portada.
2. Página de licencia, si procede.
3. Dedicatoria, opcional.
4. Agradecimientos, opcional.
5. Resumen.
6. `Summary` en inglés, recomendado.
7. Índice general.
8. Índice de figuras, si procede.
9. Índice de tablas, si procede.
10. Índice de fragmentos de código, si procede.
11. Lista de acrónimos o glosario, si procede.

### Capítulos principales

1. Introducción.
2. Objetivos y planificación.
3. Estado del arte.
4. Diseño e implementación.
5. Experimentos y validación.
6. Impacto y responsabilidad.
7. Conclusiones y trabajos futuros.
8. Bibliografía.
9. Anexos.

---

## 7. Explicación detallada de cada apartado

## 7.1. Portada

### Contenido

Debe incluir:

- Escuela de Ingeniería de Fuenlabrada;
- nombre de la titulación;
- indicación de que se trata de un Trabajo Fin de Grado;
- título;
- subtítulo, si procede;
- autor;
- tutor o tutora;
- curso académico.

### Extensión

- **Una página.**

### Recomendaciones

- Utilizar el modelo oficial de la EIF.
- Evitar títulos vagos.
- Reflejar el problema, la solución o el dominio de aplicación.
- Confirmar el título definitivo con el tutor.

---

## 7.2. Página de licencia

### Carácter

- **[OPCIONAL]**

### Cuándo incluirla

Cuando se publique la memoria o el código en abierto, o cuando resulte útil aclarar las condiciones de reutilización.

### Recomendaciones

- No conservar automáticamente la licencia de ejemplo de la plantilla.
- Distinguir entre licencia de la memoria y licencia del código.
- Confirmar la elección con el tutor cuando existan dudas.

### Extensión

- **Cero o una página.**

---

## 7.3. Dedicatoria y agradecimientos

### Carácter

- **[OPCIONAL]**

### Extensión recomendada

| Elemento | Extensión |
|---|---:|
| Dedicatoria | Una o dos líneas |
| Agradecimientos | Entre media página y una página |

### Recomendaciones

- Mantener un tono sobrio.
- No convertir esta sección en un capítulo extenso.

---

## 7.4. Resumen

### Carácter

- **[OBLIGATORIO]**

### Extensión

- Máximo oficial: **una página**.
- Orientación práctica: entre **300 y 500 palabras**, siempre que quepa en una página con la maquetación utilizada.

### Contenido

Debe sintetizar:

1. contexto y problema;
2. objetivo principal;
3. metodología;
4. tecnologías o enfoque relevantes;
5. principales resultados;
6. conclusiones.

### Recomendaciones

- Redactarlo al final del proceso.
- No introducir citas salvo necesidad excepcional.
- No incluir afirmaciones que no estén justificadas en el cuerpo de la memoria.
- No confundir resumen con introducción.

---

## 7.5. Summary

### Carácter

- **[RECOMENDADO]**

### Extensión

- Máximo orientativo: **una página**.

### Contenido

- Traducción fiel del resumen al inglés.
- Terminología técnica revisada.
- Mismo alcance y mismos resultados que la versión en castellano.

---

## 7.6. Índices

### Índice general

- **[OBLIGATORIO]**
- Debe reflejar la estructura real del documento.

### Índices adicionales

- **[RECOMENDADOS CUANDO PROCEDA]**

Incluir:

- índice de figuras si existen varias figuras;
- índice de tablas si existen varias tablas;
- índice de código si se incorporan varios fragmentos;
- glosario o acrónimos si se utilizan términos técnicos recurrentes.

### Recomendaciones

- Evitar una profundidad excesiva.
- Utilizar normalmente tres niveles como máximo: capítulo, sección y subsección.

---

## 7.7. Capítulo 1. Introducción

### Objetivo del capítulo

Presentar el problema, el contexto y el alcance del TFG. Al terminar la introducción, una persona ajena al proyecto debe entender:

- qué problema existe;
- por qué resulta relevante;
- quién se beneficia de resolverlo;
- qué parte concreta aborda el TFG;
- qué queda fuera del alcance;
- cómo se organiza el documento.

### Estructura recomendada

#### 1.1. Contexto y motivación

Explicar el ámbito general y avanzar progresivamente hacia el problema concreto.

#### 1.2. Planteamiento del problema

Definir:

- situación actual;
- dificultad concreta;
- usuarios o agentes afectados;
- consecuencias;
- limitaciones de las soluciones existentes;
- criterios que permitirían considerar útil una solución.

#### 1.3. Alcance y limitaciones iniciales

Delimitar:

- funcionalidades incluidas;
- datos contemplados;
- usuarios considerados;
- escenarios excluidos;
- restricciones temporales, legales y computacionales.

#### 1.4. Estructura de la memoria

Resumir el contenido de cada capítulo.

### Extensión orientativa

- **Entre 4 y 7 páginas.**

### Errores que deben evitarse

- Introducciones genéricas sin conexión con el problema.
- Frases promocionales sobre la inteligencia artificial.
- Explicación prematura de detalles de implementación.
- Ausencia de delimitación del alcance.
- Confusión entre motivación y objetivos.

---

## 7.8. Capítulo 2. Objetivos y planificación

### Objetivo del capítulo

Definir qué se pretende conseguir y cómo se organizó el trabajo.

### Estructura recomendada

#### 2.1. Objetivo general

Debe expresarse en una o dos frases y utilizar verbos concretos en infinitivo:

- diseñar;
- desarrollar;
- implementar;
- evaluar;
- comparar;
- validar;
- analizar.

Debe responder:

> ¿Qué resultado principal pretende alcanzar el TFG?

#### 2.2. Objetivos específicos

Deben ser:

- concretos;
- verificables;
- coherentes con el objetivo general;
- revisables al final del proyecto.

Ejemplos de tipos de objetivo:

- analizar soluciones existentes;
- recopilar y procesar datos;
- diseñar una arquitectura;
- implementar componentes;
- comparar estrategias;
- definir métricas;
- evaluar el sistema;
- documentar la solución.

#### 2.3. Requisitos y restricciones

Separar:

| Tipo | Contenido |
|---|---|
| Requisitos funcionales | Qué debe hacer el sistema |
| Requisitos no funcionales | Rendimiento, trazabilidad, reproducibilidad, seguridad, mantenibilidad |
| Restricciones | Tiempo, recursos, licencias, privacidad, disponibilidad de datos |

#### 2.4. Planificación temporal

Explicar:

- fases;
- cronograma;
- hitos;
- dependencias;
- desviaciones;
- decisiones adoptadas para corregir desviaciones.

Puede incluirse un diagrama de Gantt.

### Extensión orientativa

- **Entre 3 y 6 páginas.**

### Comprobación

Cada objetivo específico debe tener una correspondencia posterior con:

- una parte de la implementación;
- una prueba o resultado;
- una conclusión.

---

## 7.9. Capítulo 3. Estado del arte

### Objetivo del capítulo

Explicar los conceptos, tecnologías, trabajos previos y alternativas que existían antes del TFG. Debe justificar el enfoque elegido.

### Regla esencial

El estado del arte no describe el trabajo propio. Describe el contexto técnico y académico necesario para entenderlo.

### Estructura recomendada

#### 3.1. Conceptos fundamentales

Definir solo los conceptos necesarios.

#### 3.2. Tecnologías relevantes

Explicar:

- finalidad;
- funcionamiento general;
- ventajas;
- limitaciones;
- alternativas;
- relación con el proyecto.

#### 3.3. Trabajos relacionados y soluciones existentes

Revisar:

- artículos científicos;
- documentación oficial;
- repositorios relevantes;
- proyectos comparables;
- herramientas;
- soluciones comerciales cuando proceda.

#### 3.4. Limitaciones detectadas

Identificar carencias o problemas no resueltos.

#### 3.5. Posicionamiento del TFG

Aclarar:

- qué necesidad aborda;
- qué aporta;
- qué no pretende resolver.

### Extensión orientativa

- **Entre 12 y 20 páginas.**

### Recomendaciones

- Priorizar fuentes primarias.
- Evitar convertir el capítulo en un tutorial genérico.
- No enumerar tecnologías sin explicar su relevancia.
- Introducir referencias bibliográficas desde el primer borrador.

---

## 7.10. Capítulo 4. Diseño e implementación

### Objetivo del capítulo

Explicar con precisión el trabajo técnico propio.

### Regla esencial

Este es el núcleo de la memoria. Debe dejar claro:

- qué se construyó;
- cómo funciona;
- por qué se tomaron determinadas decisiones;
- qué alternativas se consideraron;
- qué limitaciones presenta.

### Estructura recomendada

#### 4.1. Metodología de desarrollo

Describir:

- fases;
- iteraciones;
- prototipos;
- decisiones;
- cambios relevantes.

#### 4.2. Arquitectura general

Incluir un diagrama de alto nivel con:

- entradas;
- módulos;
- flujo de datos;
- almacenamiento;
- dependencias;
- salidas.

#### 4.3. Componentes del sistema

Para cada componente explicar:

- finalidad;
- entradas;
- proceso;
- salidas;
- decisiones técnicas;
- limitaciones.

#### 4.4. Datos y procesamiento

Cuando proceda:

- origen;
- formato;
- limpieza;
- normalización;
- fragmentación;
- metadatos;
- almacenamiento;
- control de calidad.

#### 4.5. Decisiones técnicas

Para cada decisión relevante:

1. explicar el problema;
2. identificar alternativas;
3. definir criterios;
4. justificar la solución elegida;
5. analizar consecuencias.

#### 4.6. Implementación

Explicar:

- estructura del repositorio;
- módulos;
- librerías;
- dependencias;
- interfaces;
- configuración;
- fragmentos de código breves cuando aporten valor.

#### 4.7. Reproducibilidad

Incluir:

- requisitos;
- instalación;
- configuración;
- variables de entorno;
- comandos;
- pasos de ejecución;
- restricciones conocidas.

### Extensión orientativa

- **Entre 18 y 30 páginas.**

### Figuras y código

- Utilizar diagramas antes que grandes bloques de código.
- Explicar toda figura en el texto.
- Referenciar todas las figuras y tablas.
- Incluir únicamente código breve y relevante.
- Mantener el código completo en el repositorio.

---

## 7.11. Capítulo 5. Experimentos y validación

### Objetivo del capítulo

Demostrar si la solución funciona y analizar sus límites.

### Regla esencial

Una demostración visual no sustituye a una evaluación. No basta con enseñar que el programa devuelve resultados.

### Estructura recomendada

#### 5.1. Preguntas de evaluación

Definir qué se quiere comprobar:

- calidad;
- utilidad;
- coste;
- rendimiento;
- comparación entre configuraciones;
- robustez;
- errores.

#### 5.2. Entorno experimental

Documentar:

- hardware;
- software;
- versiones;
- modelos;
- parámetros;
- recursos computacionales;
- semillas aleatorias cuando proceda.

#### 5.3. Datos o casos de prueba

Explicar:

- origen;
- tamaño;
- selección;
- categorías;
- particiones;
- representatividad;
- sesgos;
- limitaciones.

#### 5.4. Métricas

Definir:

- qué mide cada métrica;
- cómo se calcula;
- por qué es adecuada;
- qué limitaciones tiene.

#### 5.5. Experimentos

Para cada experimento:

1. hipótesis;
2. configuración;
3. variable analizada;
4. resultados;
5. interpretación.

#### 5.6. Resultados

Presentar:

- tablas;
- gráficas;
- comparativas;
- análisis cualitativo;
- interpretación.

#### 5.7. Análisis de errores

Incluir casos problemáticos:

- errores de recuperación;
- respuestas incorrectas;
- alucinaciones;
- pérdida de contexto;
- resultados ambiguos;
- costes excesivos;
- limitaciones de datos.

#### 5.8. Limitaciones

Aclarar:

- qué conclusiones son válidas;
- qué no puede garantizarse;
- qué factores externos afectan a los resultados.

### Extensión orientativa

- **Entre 12 y 20 páginas.**

---

## 7.12. Capítulo 6. Impacto y responsabilidad

### Carácter para este TFG

- **[RECOMENDADO]**
- Debe incluirse salvo indicación contraria del tutor.

### Estructura recomendada

#### 6.1. Impacto social

Analizar:

- grupos afectados;
- accesibilidad;
- privacidad;
- seguridad;
- bienestar;
- igualdad;
- posibles usos indebidos.

#### 6.2. Impacto económico

Analizar:

- costes;
- mantenimiento;
- productividad;
- escalabilidad;
- dependencia de servicios externos;
- viabilidad.

#### 6.3. Impacto medioambiental

Analizar:

- consumo energético;
- inferencia;
- almacenamiento;
- recursos computacionales;
- optimización.

#### 6.4. Responsabilidad ética y profesional

Analizar:

- riesgos;
- trazabilidad;
- transparencia;
- propiedad intelectual;
- protección de datos;
- normativa;
- supervisión humana;
- límites de uso.

### Extensión orientativa

- **Entre 2 y 4 páginas.**

---

## 7.13. Capítulo 7. Conclusiones y trabajos futuros

### Objetivo del capítulo

Responder:

> ¿Qué se ha conseguido realmente y qué queda pendiente?

### Estructura recomendada

#### 7.1. Consecución de objetivos

Recuperar uno a uno los objetivos iniciales y explicar:

- grado de cumplimiento;
- evidencias;
- limitaciones;
- obstáculos;
- medidas adoptadas.

#### 7.2. Aplicación de conocimientos del grado

Explicar qué conocimientos o capacidades del grado se aplicaron y cómo.

No limitarse a enumerar asignaturas. Relacionar conocimientos con decisiones concretas.

#### 7.3. Competencias adquiridas

Explicar nuevos aprendizajes:

- tecnologías;
- arquitectura;
- evaluación;
- ingeniería de software;
- gestión de datos;
- planificación;
- resolución de problemas;
- comunicación técnica.

#### 7.4. Trabajos futuros

Proponer mejoras realistas:

- ampliaciones;
- funcionalidades;
- experimentos;
- optimizaciones;
- despliegue;
- nuevas fuentes de datos;
- mejoras de usabilidad;
- refactorización.

### Extensión orientativa

- **Entre 4 y 7 páginas.**

---

## 7.14. Bibliografía

### Carácter

- **[OBLIGATORIO]**

### Recomendaciones

- Gestionar referencias mediante `memoria.bib`.
- Mantener un único estilo.
- Citar todas las ideas, datos, diagramas adaptados y contenidos ajenos.
- Priorizar:
  1. artículos científicos;
  2. documentación oficial;
  3. normativa;
  4. documentación técnica primaria;
  5. repositorios oficiales;
  6. fuentes secundarias solo cuando aporten valor.
- Evitar referencias inventadas.
- Verificar que toda cita utilizada aparece en la bibliografía.
- Eliminar entradas no citadas.
- Añadir DOI, URL y fecha de consulta cuando proceda.

### Regla para el asistente

Nunca inventar autores, títulos, años, DOI, enlaces o resultados. Si una referencia no se ha comprobado, marcarla como:

`[REFERENCIA PENDIENTE DE VERIFICAR]`

---

## 7.15. Anexos

### Carácter

- **[CUANDO SEAN NECESARIOS]**

### Posibles contenidos

- manual de instalación;
- manual de usuario;
- enlace al repositorio;
- estructura completa de carpetas;
- configuraciones;
- variables de entorno sin secretos;
- prompts extensos;
- resultados adicionales;
- tablas completas;
- ejemplos detallados;
- diagramas complementarios;
- licencias;
- instrucciones de despliegue;
- documentación de datasets.

### Regla esencial

La memoria principal debe ser comprensible sin depender constantemente de los anexos.

### Extensión

- No computa dentro del máximo oficial de 100 páginas.
- Debe mantenerse una selección razonable y útil.

---

## 8. Distribución orientativa de páginas

La única extensión global oficial es el máximo de 100 páginas sin anexos. La siguiente tabla es una referencia práctica:

| Bloque | Extensión orientativa |
|---|---:|
| Portada | 1 página |
| Licencia | 0–1 página |
| Dedicatoria | 0–1 página |
| Agradecimientos | 0–1 página |
| Resumen | Máximo 1 página |
| Summary | Máximo orientativo 1 página |
| Índices | Variable |
| Introducción | 4–7 páginas |
| Objetivos y planificación | 3–6 páginas |
| Estado del arte | 12–20 páginas |
| Diseño e implementación | 18–30 páginas |
| Experimentos y validación | 12–20 páginas |
| Impacto y responsabilidad | 2–4 páginas |
| Conclusiones y trabajos futuros | 4–7 páginas |
| Bibliografía | Variable |
| Anexos | Sin límite oficial específico |

### Objetivo práctico

Una memoria principal de aproximadamente **60–75 páginas**, sin anexos, puede ser adecuada para un TFG técnico bien desarrollado. No debe añadirse contenido de relleno para alcanzar una cifra artificial.

---

## 9. Adaptación recomendada al sistema RAG jurídico

La siguiente estructura puede utilizarse como punto de partida para el proyecto concreto.

### 1. Introducción

1.1. Contexto y motivación  
1.2. Dificultad de consultar normativa extensa y cambiante  
1.3. Planteamiento del problema  
1.4. Alcance y limitaciones  
1.5. Estructura de la memoria  

### 2. Objetivos y planificación

2.1. Objetivo general  
2.2. Objetivos específicos  
2.3. Requisitos funcionales  
2.4. Requisitos no funcionales  
2.5. Restricciones  
2.6. Planificación temporal  

### 3. Estado del arte

3.1. Recuperación de información  
3.2. Modelos de lenguaje  
3.3. Limitaciones de los LLM: alucinaciones, actualización y trazabilidad  
3.4. Arquitectura Retrieval-Augmented Generation (RAG)  
3.5. Recuperación léxica  
3.6. Recuperación semántica  
3.7. Recuperación híbrida  
3.8. Reordenación de resultados  
3.9. Embeddings  
3.10. Fragmentación documental  
3.11. Índices y bases vectoriales  
3.12. Generación fundamentada y citas  
3.13. Evaluación de sistemas RAG  
3.14. Particularidades del dominio jurídico  
3.15. Posicionamiento del proyecto  

### 4. Diseño e implementación

4.1. Arquitectura general  
4.2. Fuentes documentales  
4.3. Obtención de normativa  
4.4. Limpieza y normalización  
4.5. Gestión de metadatos  
4.6. Fragmentación documental  
4.7. Generación de embeddings  
4.8. Indexación  
4.9. Recuperación léxica  
4.10. Recuperación semántica  
4.11. Recuperación híbrida  
4.12. Reordenación, si procede  
4.13. Construcción del contexto  
4.14. Generación de respuestas  
4.15. Citas y trazabilidad  
4.16. Organización del repositorio  
4.17. Reproducibilidad  

### 5. Experimentos y validación

5.1. Preguntas de evaluación  
5.2. Dataset de consultas  
5.3. Criterios de anotación  
5.4. Métricas de recuperación  
5.5. Métricas o rúbrica de generación  
5.6. Línea base  
5.7. Comparación de recuperación léxica, semántica e híbrida  
5.8. Efecto del tamaño de fragmento  
5.9. Efecto de `top-k`  
5.10. Evaluación de citas  
5.11. Análisis de errores  
5.12. Limitaciones  

### 6. Impacto y responsabilidad

6.1. Acceso a información jurídica  
6.2. Riesgo de respuestas incorrectas  
6.3. Privacidad  
6.4. Trazabilidad  
6.5. Supervisión humana  
6.6. Coste computacional  
6.7. Propiedad intelectual y licencias  
6.8. Límites de uso: el sistema no sustituye asesoramiento jurídico profesional  

### 7. Conclusiones y trabajos futuros

7.1. Consecución de objetivos  
7.2. Principales resultados  
7.3. Limitaciones  
7.4. Conocimientos del grado aplicados  
7.5. Competencias adquiridas  
7.6. Trabajos futuros  

### Anexos

A. Instalación  
B. Ejecución  
C. Repositorio  
D. Configuración  
E. Prompts  
F. Ejemplos completos  
G. Resultados adicionales  
H. Licencias  

---

## 10. Reglas de redacción académica

El asistente debe aplicar estas reglas al redactar o revisar texto.

### 10.1. Precisión

- Evitar afirmaciones amplias sin evidencia.
- Sustituir frases promocionales por formulaciones verificables.
- Delimitar el alcance de cada afirmación.
- Diferenciar resultados, hipótesis e interpretaciones.

### 10.2. Coherencia

Cada capítulo debe cumplir una función:

| Capítulo | Función |
|---|---|
| Introducción | Definir el problema |
| Objetivos | Indicar qué se pretende conseguir |
| Estado del arte | Explicar lo existente |
| Diseño e implementación | Describir la contribución propia |
| Experimentos | Evaluar la solución |
| Impacto | Analizar consecuencias y responsabilidades |
| Conclusiones | Valorar los resultados y cerrar el trabajo |

### 10.3. Trazabilidad

- No introducir datos sin fuente.
- No introducir decisiones técnicas sin justificación.
- No presentar resultados sin explicar el experimento.
- No presentar una conclusión que no se derive de resultados anteriores.

### 10.4. Estilo

- Utilizar español académico claro.
- Evitar expresiones coloquiales.
- Evitar frases innecesariamente largas.
- Reducir repeticiones.
- Mantener consistencia terminológica.
- Definir acrónimos la primera vez:
  - `Retrieval-Augmented Generation (RAG)`.
- Revisar ortografía y puntuación.

### 10.5. Tablas, figuras y gráficas

Toda tabla, figura o gráfica debe:

1. tener número;
2. tener título o pie;
3. mencionarse en el texto;
4. explicarse;
5. indicar la fuente si no es propia;
6. contribuir a una idea concreta.

### 10.6. Código

- Evitar grandes bloques.
- Incluir únicamente fragmentos relevantes.
- Explicar su finalidad.
- Remitir al repositorio para el código completo.
- No incluir credenciales ni secretos.

---

## 13. Lista de comprobación global

### Requisitos formales

- [ ] La memoria principal no supera 100 páginas sin anexos.
- [ ] El documento utiliza DIN A4 y un formato legible.
- [ ] La portada sigue el modelo oficial.
- [ ] El resumen no supera una página.
- [ ] El PDF compila correctamente.
- [ ] Se ha confirmado el idioma.
- [ ] Se han eliminado restos de la plantilla.
- [ ] No aparecen secretos ni credenciales.

### Estructura

- [ ] Existe índice general.
- [ ] La introducción plantea un problema concreto.
- [ ] El alcance está delimitado.
- [ ] El objetivo general es claro.
- [ ] Los objetivos específicos son verificables.
- [ ] Los requisitos están definidos cuando procede.
- [ ] El estado del arte cita fuentes.
- [ ] El trabajo propio se distingue de lo ajeno.
- [ ] La arquitectura se explica.
- [ ] Las decisiones técnicas están justificadas.
- [ ] Existen experimentos.
- [ ] Los resultados se interpretan.
- [ ] Existe análisis de errores.
- [ ] Existen limitaciones.
- [ ] Se incluye impacto y responsabilidad.
- [ ] Las conclusiones revisan los objetivos.
- [ ] Se mencionan competencias aplicadas.
- [ ] Se mencionan competencias adquiridas.
- [ ] Se incluyen trabajos futuros.

### Tablas, figuras y código

- [ ] Todas las figuras están numeradas.
- [ ] Todas las figuras se mencionan y explican.
- [ ] Todas las tablas están numeradas.
- [ ] Todas las tablas se interpretan.
- [ ] Las gráficas tienen ejes y unidades.
- [ ] Las fuentes ajenas están citadas.
- [ ] El código incluido es breve y útil.
- [ ] El código completo se remite al repositorio.

### Bibliografía

- [ ] No existen referencias inventadas.
- [ ] Toda cita aparece en la bibliografía.
- [ ] No hay entradas bibliográficas sin utilizar.
- [ ] Se priorizan fuentes primarias.
- [ ] Los datos bibliográficos están completos.
- [ ] Se han revisado DOI, URL y fechas cuando proceda.

### Anexos

- [ ] El enlace al repositorio aparece claramente.
- [ ] Se incluye instalación.
- [ ] Se incluye ejecución.
- [ ] Se documentan configuraciones relevantes.
- [ ] No se incluyen secretos.
- [ ] Los anexos aportan valor y no sustituyen la explicación principal.

---

## 14. Fuentes de referencia verificadas

### Normativa oficial

**Reglamento específico del Trabajo Fin de Grado de la Escuela de Ingeniería de Fuenlabrada (EIF), URJC.**  
Documento público consultado:  
`https://www.urjc.es/images/facultades/eif/documentos/Reglamento_TFG_EIF_10_25.pdf`

### Página de estudiantes de la EIF

Incluye documentación y convocatorias relacionadas con el TFG:  
`https://www.urjc.es/estudiantes-eif`

### Plataforma de TFG de la URJC

`https://servicios.urjc.es/tfg/`

### Plantilla LaTeX utilizada

Repositorio actualizado:  
`https://github.com/glimmerphoenix/plantilla-memoria-TFG-TFM`

### Advertencia

Antes de la entrega final debe comprobarse si la EIF ha publicado una versión posterior del reglamento, nuevas instrucciones, cambios de convocatoria o indicaciones adicionales.

---

## 15. Instrucción final para el asistente

Utiliza este documento como marco permanente. Ayuda a construir una memoria rigurosa, verificable y técnicamente sólida. No rellenes vacíos con suposiciones. No conviertas recomendaciones en obligaciones. No inventes referencias ni resultados. Cuando falte información, señálalo de forma explícita. Cuando exista una decisión abierta, propón alternativas y explica sus consecuencias.
