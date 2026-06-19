"""Tests del analizador léxico español (offline, deterministas)."""

from src.retrieval.text_analysis import SpanishAnalyzer


def test_conserva_cifras_con_separador_de_millares() -> None:
    tokens = SpanishAnalyzer().analyze("El contrato menor de obras llega hasta 40.000 euros")
    assert "40.000" in tokens  # la cifra es señal léxica clave en legal


def test_elimina_stopwords() -> None:
    assert SpanishAnalyzer(stem=False).analyze("el de la y los con para") == []


def test_stemming_agrupa_singular_y_plural() -> None:
    analyzer = SpanishAnalyzer()
    assert analyzer.analyze("contrato") == analyzer.analyze("contratos")


def test_sin_stemming_distingue_morfologia() -> None:
    analyzer = SpanishAnalyzer(stem=False)
    assert analyzer.analyze("contrato") != analyzer.analyze("contratos")


def test_es_determinista() -> None:
    analyzer = SpanishAnalyzer()
    assert analyzer.analyze("Plazo de subvención") == analyzer.analyze("Plazo de subvención")


def test_min_token_len_no_descarta_cifras_cortas() -> None:
    tokens = SpanishAnalyzer(stem=False, min_token_len=4).analyze("IVA del 5 por ciento")
    assert "5" in tokens  # un umbral como "5" se conserva pese a min_token_len
    assert "iva" not in tokens  # "iva" (3 < 4) sí se descarta


def test_signature_refleja_la_configuracion() -> None:
    sig = SpanishAnalyzer(stem=False, remove_stopwords=False, min_token_len=3).signature()
    assert sig == {
        "language": "es",
        "remove_stopwords": False,
        "stem": False,
        "min_token_len": 3,
    }
