"""Análisis léxico de texto en español para el índice BM25 (tokenización + stopwords + stemming).

Componente **propio y determinista** (el algoritmo de scoring se delega en `rank_bm25`). La
tokenización es específica del dominio legal: **conserva las cifras con separador de millares
(`40.000`) y decimales (`1,5`)** —que son justo donde el recuperador léxico aporta frente al denso—,
elimina stopwords del español con una lista curada y aplica el stemmer Snowball.

`SpanishAnalyzer.analyze` es **puro** (mismo texto → mismos tokens). `signature()` expone la
configuración para registrarla en los reports y que la comparación denso vs léxico sea reproducible.
"""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

# Un token es una secuencia numérica (con separador de millares/decimal) o una palabra en español.
# Las cifras se conservan enteras ("40.000", "1,5") porque son señal léxica de alto valor en legal.
_TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[a-záéíóúüñ]+")

# Stopwords del español (artículos, preposiciones, conjunciones, pronombres, auxiliares frecuentes).
# Se mantienen las formas acentuadas: el corpus del BOE y las preguntas vienen tildados.
SPANISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "al",
        "algo",
        "algunas",
        "algunos",
        "ante",
        "antes",
        "como",
        "con",
        "contra",
        "cual",
        "cuales",
        "cuando",
        "de",
        "del",
        "desde",
        "donde",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ello",
        "ellos",
        "en",
        "entre",
        "era",
        "eran",
        "es",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estaba",
        "estado",
        "estamos",
        "estan",
        "están",
        "estar",
        "estas",
        "este",
        "esto",
        "estos",
        "fue",
        "fueron",
        "ha",
        "habia",
        "había",
        "han",
        "hasta",
        "hay",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "más",
        "mas",
        "me",
        "mi",
        "mis",
        "mucho",
        "muy",
        "nada",
        "ni",
        "no",
        "nos",
        "nosotros",
        "o",
        "os",
        "otra",
        "otras",
        "otro",
        "otros",
        "para",
        "pero",
        "poco",
        "por",
        "porque",
        "que",
        "qué",
        "quien",
        "quienes",
        "se",
        "sea",
        "ser",
        "si",
        "sí",
        "sin",
        "sobre",
        "son",
        "su",
        "sus",
        "tal",
        "también",
        "tanto",
        "te",
        "tiene",
        "tienen",
        "todo",
        "todos",
        "tu",
        "tus",
        "un",
        "una",
        "uno",
        "unos",
        "y",
        "ya",
        "yo",
    }
)


class SpanishAnalyzer:
    """Tokeniza, filtra stopwords y aplica stemming Snowball. Reutilizable y determinista.

    Parámetros como knobs del experimento (quedan en `signature()`):
    - `remove_stopwords`: elimina palabras vacías antes del stemming.
    - `stem`: reduce las palabras a su raíz (mejora el recall morfológico; las cifras no se tocan).
    - `min_token_len`: descarta tokens **alfabéticos** más cortos (las cifras se conservan siempre,
      porque un umbral como "5" es significativo).
    """

    def __init__(
        self, *, remove_stopwords: bool = True, stem: bool = True, min_token_len: int = 2
    ) -> None:
        self._remove_stopwords = remove_stopwords
        self._stem = stem
        self._min_token_len = min_token_len
        self._stopwords = SPANISH_STOPWORDS if remove_stopwords else frozenset()
        self._stemmer = snowballstemmer.stemmer("spanish") if stem else None

    def analyze(self, text: str) -> list[str]:
        """Texto → lista de tokens normalizados (minúsculas, sin stopwords y con stemming)."""
        if not text:
            return []
        normalized = unicodedata.normalize("NFC", text).lower()
        tokens = _TOKEN_RE.findall(normalized)
        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        tokens = [t for t in tokens if t[0].isdigit() or len(t) >= self._min_token_len]
        if self._stemmer is not None:
            tokens = [t if t[0].isdigit() else self._stemmer.stemWord(t) for t in tokens]
        return tokens

    def signature(self) -> dict:
        """Configuración del analizador, para registrar la corrida y reproducirla."""
        return {
            "language": "es",
            "remove_stopwords": self._remove_stopwords,
            "stem": self._stem,
            "min_token_len": self._min_token_len,
        }
