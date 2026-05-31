"""Descarga la respuesta raw de una norma BOE y genera su manifest.

Uso:
    uv run python scripts/download_boe_raw.py BOE-A-2015-10565

Llama a la API externa del BOE. Guarda los XML en `data/raw/boe/<norm_id>/` y el
manifest en `data/manifests/<norm_id>.json`. Devuelve exit code != 0 si falla.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite `import src...` al ejecutar el script directamente (la raíz del repo no
# está en sys.path cuando se lanza `python scripts/download_boe_raw.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.boe.client import BoeClient  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.core.exceptions import BoeApiError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Uso: python scripts/download_boe_raw.py <BOE-ID>\n"
            "Ejemplo: python scripts/download_boe_raw.py BOE-A-2015-10565",
            file=sys.stderr,
        )
        return 2

    norm_id = argv[0]
    settings = get_settings()

    try:
        with BoeClient(base_url=settings.boe_api_base) as client:
            downloaded = client.download_norm_raw(norm_id)
            manifest_path = client.write_manifest(norm_id, downloaded)
    except (BoeApiError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Norma descargada: {norm_id}")
    for endpoint_name, file_path in downloaded.items():
        print(f"  {endpoint_name}: {file_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
