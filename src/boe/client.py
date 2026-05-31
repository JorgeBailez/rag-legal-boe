"""Cliente de la API de legislación consolidada del BOE.

Responsabilidad: descargar una norma por identificador (BOE-A-YYYY-NNNNN), sus
metadatos, análisis, texto e índice de bloques; gestionar errores de red/HTTP y
guardar respuestas raw versionadas por identificador, con un manifest reproducible.

Esta capa NO parsea contenido: guarda bytes tal cual y delega el parsing a fases
posteriores. Mantiene la ingesta desacoplada del resto del pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.core.exceptions import BoeApiError

# Endpoints de legislación consolidada usados por la tarea de ingesta raw.
# El nombre lógico (clave) determina el fichero de salida `<nombre>.xml`.
ENDPOINTS: dict[str, str] = {
    "full": "/legislacion-consolidada/id/{id}",
    "metadatos": "/legislacion-consolidada/id/{id}/metadatos",
    "analisis": "/legislacion-consolidada/id/{id}/analisis",
    "metadata_eli": "/legislacion-consolidada/id/{id}/metadata-eli",
    "texto": "/legislacion-consolidada/id/{id}/texto",
    "indice": "/legislacion-consolidada/id/{id}/texto/indice",
}

# Validación básica del identificador BOE: BOE-A-YYYY-NNNNN (sin sobre-complicar).
NORM_ID_RE = re.compile(r"^BOE-[A-Z]-\d{4}-\d+$")

SOURCE = "BOE legislación consolidada"


class BoeClient:
    """Cliente mínimo para descargar y persistir una norma BOE en crudo."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        raw_dir: Path = Path("data/raw/boe"),
        manifest_dir: Path = Path("data/manifests"),
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.raw_dir = Path(raw_dir)
        self.manifest_dir = Path(manifest_dir)

        if client is None:
            self._client = httpx.Client(
                timeout=timeout,
                headers={"Accept": "application/xml"},
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    # -- Ciclo de vida -----------------------------------------------------

    def close(self) -> None:
        """Cierra el cliente HTTP solo si fue creado por esta instancia."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> BoeClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- Utilidades --------------------------------------------------------

    def build_url(self, path: str) -> str:
        """Compone la URL absoluta a partir de la base y una ruta relativa."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _validate_norm_id(norm_id: str) -> None:
        """Valida la forma básica del identificador; lanza ValueError si no casa."""
        if not NORM_ID_RE.match(norm_id):
            raise ValueError(
                f"Identificador BOE inválido: {norm_id!r} (esperado formato BOE-A-YYYY-NNNNN)"
            )

    # -- Descarga ----------------------------------------------------------

    def fetch_path(self, path: str) -> bytes:
        """Descarga una ruta relativa y devuelve el contenido en bytes.

        Cualquier status no exitoso, timeout o error de red se traduce a
        `BoeApiError` incluyendo el endpoint afectado.
        """
        url = self.build_url(path)
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BoeApiError(
                f"Respuesta HTTP {exc.response.status_code} en endpoint {path}"
            ) from exc
        except httpx.RequestError as exc:
            raise BoeApiError(f"Error de red al solicitar endpoint {path}: {exc}") from exc
        return response.content

    def fetch_norm(self, norm_id: str) -> dict[str, bytes]:
        """Descarga todos los endpoints de una norma y devuelve sus bytes."""
        self._validate_norm_id(norm_id)
        return {name: self.fetch_path(path.format(id=norm_id)) for name, path in ENDPOINTS.items()}

    # -- Persistencia ------------------------------------------------------

    def save_raw_response(self, norm_id: str, endpoint_name: str, content: bytes) -> Path:
        """Guarda el contenido raw como `<endpoint_name>.xml` bajo `raw_dir/norm_id`."""
        out_dir = self.raw_dir / norm_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{endpoint_name}.xml"
        out_path.write_bytes(content)
        return out_path

    def download_norm_raw(
        self,
        norm_id: str,
        optional_endpoints: frozenset[str] = frozenset(),
    ) -> dict[str, Path]:
        """Descarga y guarda los endpoints de la norma; devuelve los efectivamente guardados.

        Los endpoints listados en `optional_endpoints` que fallen (p. ej. 404) se omiten;
        un fallo en un endpoint obligatorio re-lanza `BoeApiError`.
        """
        self._validate_norm_id(norm_id)
        saved: dict[str, Path] = {}
        for name, path in ENDPOINTS.items():
            try:
                content = self.fetch_path(path.format(id=norm_id))
            except BoeApiError:
                if name in optional_endpoints:
                    continue
                raise
            saved[name] = self.save_raw_response(norm_id, name, content)
        return saved

    def write_manifest(self, norm_id: str, downloaded_files: dict[str, Path]) -> Path:
        """Genera el manifest JSON con sha256 y size_bytes de cada fichero."""
        files = []
        for endpoint_name, file_path in downloaded_files.items():
            data = file_path.read_bytes()
            files.append(
                {
                    "endpoint_name": endpoint_name,
                    "path": file_path.as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )

        manifest = {
            "norm_id": norm_id,
            "source": SOURCE,
            "base_url": self.base_url,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "files": files,
        }

        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / f"{norm_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_path
