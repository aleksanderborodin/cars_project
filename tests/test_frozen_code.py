"""A later workspace edit must not change a long campaign's imported code."""
import subprocess
import sys

import pytest

from cars.research.frozen_code import freeze_sources, source_hashes


def test_frozen_import_survives_live_edit_and_detects_tampering(tmp_path):
    repository = tmp_path / "repository"
    package = repository / "src" / "campaign_probe"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("value = 17\n")
    for name in ("pyproject.toml", "uv.lock"):
        (repository / name).write_text("# synthetic fixture\n")
    expected = source_hashes(repository)
    frozen = tmp_path / "snapshot"
    environment = freeze_sources(repository, frozen, expected)
    module.write_text("value = 99\n")
    result = subprocess.check_output([sys.executable, "-c", "import campaign_probe; print(campaign_probe.value)"],
                                     env=environment, cwd=tmp_path, text=True)
    assert result.strip() == "17"
    assert source_hashes(frozen) == expected
    with pytest.raises(ValueError, match="changed before"):
        freeze_sources(repository, tmp_path / "bad_snapshot", expected)
