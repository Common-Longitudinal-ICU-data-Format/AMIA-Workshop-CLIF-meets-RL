# utils/

Shared Python helpers.

| File | Purpose |
|---|---|
| `config.py` | `load_config()` — returns the parquet directory + timezone, reading `config/config.json` if present, otherwise falling back to the bundled CLIF demo dataset that ships with `clifpy`. |

The notebooks in `code/` inline the same loader logic, so they remain
self-contained on Google Colab (where `utils/` is not available unless the
repo has been cloned into the Colab session). This module is here for
scripts and for re-use in derived projects.
