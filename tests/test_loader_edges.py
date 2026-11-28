import json

import pytest

from herhim.catalog.loader import load_manifest
from herhim.errors import PersonaError


def test_manifest_must_be_mapping(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(PersonaError, match="not a mapping"):
        load_manifest("her", root=tmp_path)
