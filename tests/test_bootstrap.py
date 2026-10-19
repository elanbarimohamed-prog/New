import importlib
import sys
import subprocess
from pathlib import Path

from app_bootstrap import refresh_package_sources


def test_bootstrap_refreshes_old_imports_and_reuses_unchanged_source(tmp_path, monkeypatch):
    package_name = "bootstrap_example"
    folder = tmp_path / package_name
    folder.mkdir()
    (folder / "__init__.py").write_text("", encoding="utf-8")
    model = folder / "models.py"
    model.write_text("class ExistingModel: pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    old = importlib.import_module(package_name + ".models")
    assert not hasattr(old, "NewCollection")
    model.write_text("class ExistingModel: pass\nclass NewCollection: pass\n", encoding="utf-8")
    clears = []
    try:
        assert refresh_package_sources(folder, package_name, lambda: clears.append(True))
        fresh = importlib.import_module(package_name + ".models")
        assert hasattr(fresh, "NewCollection")
        assert fresh is not old
        assert not refresh_package_sources(folder, package_name, lambda: clears.append(True))
        assert clears == [True]
        assert importlib.import_module(package_name + ".models") is fresh
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def test_bootstrap_does_not_clear_resources_on_initial_import(tmp_path, monkeypatch):
    package_name = "bootstrap_fresh_example"
    folder = tmp_path / package_name
    folder.mkdir()
    (folder / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    clears = []
    try:
        assert not refresh_package_sources(folder, package_name, lambda: clears.append(True))
        assert clears == []
    finally:
        sys.modules.pop(package_name, None)


def test_streamlit_recovers_a_preupdate_cached_models_module():
    project = Path(__file__).parents[1]
    script = """
import app
import app.core.models as models
delattr(models, 'NewsCollection')
delattr(app, '_sentiment_source_fingerprint')
from streamlit.testing.v1 import AppTest
test = AppTest.from_file('streamlit_app.py').run(timeout=15)
assert not test.exception, test.exception
from app.core.models import NewsCollection
assert NewsCollection
print('Recovered cached pre-update imports.')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=project,
                            text=True, capture_output=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
