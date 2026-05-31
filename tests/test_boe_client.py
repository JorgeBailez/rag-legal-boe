"""Tests unitarios del cliente BOE (sin llamadas reales a la red).

Se usa `httpx.MockTransport` para simular las respuestas de la API. Las rutas de
salida apuntan a `tmp_path`, de modo que ningún test toca `data/` real.
"""

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from src.boe.client import ENDPOINTS, BoeClient
from src.core.exceptions import BoeApiError

BASE_URL = "https://example.test/datosabiertos/api"
NORM_ID = "BOE-A-2015-10565"


def make_client(
    handler,
    tmp_path: Path,
    base_url: str = BASE_URL,
) -> BoeClient:
    """Crea un BoeClient con transporte simulado y salidas en tmp_path."""
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return BoeClient(
        base_url=base_url,
        client=http_client,
        raw_dir=tmp_path / "raw",
        manifest_dir=tmp_path / "manifests",
    )


def test_build_url_composes_relative_paths(tmp_path: Path) -> None:
    client = make_client(lambda req: httpx.Response(200), tmp_path)
    assert (
        client.build_url("/legislacion-consolidada/id/BOE-A-2015-10565")
        == f"{BASE_URL}/legislacion-consolidada/id/BOE-A-2015-10565"
    )
    # Sin doble barra aunque la base termine en "/" y el path empiece sin "/".
    client_trailing = make_client(
        lambda req: httpx.Response(200), tmp_path, base_url=BASE_URL + "/"
    )
    assert client_trailing.build_url("foo/bar") == f"{BASE_URL}/foo/bar"


def test_fetch_path_returns_bytes_on_200(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<root/>")

    client = make_client(handler, tmp_path)
    content = client.fetch_path("/legislacion-consolidada/id/BOE-A-2015-10565")
    assert content == b"<root/>"


def test_fetch_path_raises_on_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = make_client(handler, tmp_path)
    with pytest.raises(BoeApiError) as exc_info:
        client.fetch_path("/legislacion-consolidada/id/BOE-A-9999-99999")
    message = str(exc_info.value)
    assert "404" in message
    assert "/legislacion-consolidada/id/BOE-A-9999-99999" in message


def test_fetch_path_raises_on_network_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout simulado", request=request)

    client = make_client(handler, tmp_path)
    with pytest.raises(BoeApiError):
        client.fetch_path("/legislacion-consolidada/id/BOE-A-2015-10565")


def test_download_norm_raw_saves_expected_files(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<root/>")

    client = make_client(handler, tmp_path)
    downloaded = client.download_norm_raw(NORM_ID)

    assert set(downloaded) == set(ENDPOINTS)
    for endpoint_name, file_path in downloaded.items():
        assert file_path.name == f"{endpoint_name}.xml"
        assert file_path.parent == tmp_path / "raw" / NORM_ID
        assert file_path.read_bytes() == b"<root/>"


def test_write_manifest_generates_valid_json(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<root/>")

    client = make_client(handler, tmp_path)
    downloaded = client.download_norm_raw(NORM_ID)
    manifest_path = client.write_manifest(NORM_ID, downloaded)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["norm_id"] == NORM_ID
    assert manifest["base_url"] == BASE_URL
    assert "downloaded_at" in manifest
    assert len(manifest["files"]) == len(ENDPOINTS)

    expected_sha = hashlib.sha256(b"<root/>").hexdigest()
    for entry in manifest["files"]:
        assert entry["endpoint_name"] in ENDPOINTS
        assert entry["sha256"] == expected_sha
        assert entry["size_bytes"] == len(b"<root/>")
        assert entry["path"].endswith(f"{entry['endpoint_name']}.xml")


def test_invalid_norm_id_raises_value_error(tmp_path: Path) -> None:
    client = make_client(lambda req: httpx.Response(200), tmp_path)
    with pytest.raises(ValueError):
        client.fetch_norm("no-es-un-id-valido")


def test_download_norm_raw_tolerates_optional_endpoint_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # El endpoint metadata-eli devuelve 404; el resto, 200.
        if request.url.path.endswith("/metadata-eli"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"<root/>")

    client = make_client(handler, tmp_path)
    downloaded = client.download_norm_raw(NORM_ID, optional_endpoints=frozenset({"metadata_eli"}))

    assert "metadata_eli" not in downloaded  # opcional ausente, no se guarda
    assert set(downloaded) == set(ENDPOINTS) - {"metadata_eli"}


def test_download_norm_raw_raises_on_mandatory_endpoint_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/texto"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"<root/>")

    client = make_client(handler, tmp_path)
    with pytest.raises(BoeApiError):
        client.download_norm_raw(NORM_ID, optional_endpoints=frozenset({"metadata_eli"}))
