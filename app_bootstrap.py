"""Keep imported project modules in sync with source updates on long-lived hosts."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import threading

_reload_lock = threading.RLock()


def source_fingerprint(package_dir: Path):
    package_dir = Path(package_dir).resolve()
    digest = hashlib.sha256()
    for source in sorted(package_dir.rglob("*.py")):
        digest.update(source.relative_to(package_dir).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def refresh_package_sources(package_dir: Path, package_name="app", on_reload=None):
    """Reload a previously imported package only after its source fingerprint changes.

    Streamlit can rerun the entrypoint after a git pull while sys.modules still
    holds the earlier package. Do this before importing any project classes.
    External dependencies are left alone; cached resources are cleared only on
    a detected package update.
    """
    package_dir = Path(package_dir).resolve()
    fingerprint = source_fingerprint(package_dir)
    with _reload_lock:
        package = sys.modules.get(package_name)
        changed = package is not None and getattr(
            package, "_sentiment_source_fingerprint", None) != fingerprint
        if changed:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
            if on_reload:
                on_reload()
        package = importlib.import_module(package_name)
        package._sentiment_source_fingerprint = fingerprint
        return changed
