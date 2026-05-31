"""Jerarquía de excepciones del proyecto RAG Legal BOE.

Excepción base común para poder capturar cualquier error propio del dominio sin
atrapar errores ajenos, con subclases por área de responsabilidad.
"""


class RagLegalBoeError(Exception):
    """Excepción base de todo el proyecto."""


class ConfigurationError(RagLegalBoeError):
    """Configuración inválida o incompleta (settings/env)."""


class ExternalServiceError(RagLegalBoeError):
    """Fallo al comunicarse con un servicio externo (red, HTTP, timeout)."""


class BoeApiError(ExternalServiceError):
    """Error específico de la API del BOE (respuesta no válida, id inexistente)."""


class ParsingError(RagLegalBoeError):
    """Error al parsear XML/HTML del BOE al modelo documental propio."""
