"""Test del modo incremental (--only-new) de build_corpus: salta lo ya descargado, sin red."""

import json
from pathlib import Path

import pytest

import scripts.build_corpus as bc


class _FakeClient:
    """Cliente BOE falso: registra qué normas se intentan descargar (sin red)."""

    def __init__(self, **_kwargs) -> None:
        self.downloaded: list[str] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def download_norm_raw(self, norm_id: str, optional_endpoints=None) -> dict:  # noqa: ANN001
        self.downloaded.append(norm_id)
        return {}

    def write_manifest(self, norm_id: str, downloaded: dict) -> None:
        pass


def test_only_new_salta_normas_con_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "BOE-A-2015-10565.json").write_text("{}", encoding="utf-8")  # ya descargada
    report_path = tmp_path / "verification_report.json"
    report_path.write_text(
        json.dumps([{"norm_id": "BOE-A-2015-10565", "meets_criteria": True, "reasons": []}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(bc, "MANIFEST_DIR", manifests)
    monkeypatch.setattr(bc, "REPORT_PATH", report_path)
    monkeypatch.setattr(
        bc,
        "load_seed_corpus",
        lambda: [{"norm_id": "BOE-A-2015-10565"}, {"norm_id": "BOE-A-9999-1"}],
    )
    fake = _FakeClient()
    monkeypatch.setattr(bc, "BoeClient", lambda **_kw: fake)
    monkeypatch.setattr(bc, "get_settings", lambda: type("S", (), {"boe_api_base": "x"})())
    # la norma nueva no cumple criterios → evita _process (y la red real).
    monkeypatch.setattr(
        bc,
        "verify_norm",
        lambda nid, raw, dl: {"norm_id": nid, "meets_criteria": False, "reasons": ["test"]},
    )

    rc = bc.main(only_new=True)

    assert rc == 0
    assert fake.downloaded == ["BOE-A-9999-1"]  # solo la nueva se descarga
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {r["norm_id"] for r in report} == {
        "BOE-A-2015-10565",
        "BOE-A-9999-1",
    }  # vieja conservada
