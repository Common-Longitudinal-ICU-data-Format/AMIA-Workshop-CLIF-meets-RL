"""Workshop config loader.

Behaviour:
- If `config/config.json` exists at the repo root and `use_demo` is not true,
  the notebooks use the parquet directory listed in `tables_path`.
- Otherwise (no config.json, or `use_demo: true`), they fall back to the CLIF
  demo dataset that ships inside the `clifpy` package (~3 MB, open access).

This makes the same notebook runnable on Google Colab (no config.json, demo path)
and on a local clone with a `config/config.json` pointing at full CLIF-MIMIC
or a site-specific CLIF directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_config() -> Dict[str, str]:
    """Return a dict with keys: site_name, tables_path, file_type, timezone, source."""
    import clifpy

    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "config" / "config.json"

    if config_path.exists():
        with open(config_path) as fh:
            cfg = json.load(fh)
        if not cfg.get("use_demo", False):
            return {
                "site_name": cfg.get("site_name", "your_site"),
                "tables_path": cfg["tables_path"],
                "file_type": cfg.get("file_type", "parquet"),
                "timezone": cfg.get("timezone", "US/Eastern"),
                "source": str(config_path),
            }

    return {
        "site_name": "clif_demo",
        "tables_path": str(Path(clifpy.__file__).parent / "data" / "clif_demo"),
        "file_type": "parquet",
        "timezone": "US/Eastern",
        "source": "bundled_clifpy_demo",
    }
