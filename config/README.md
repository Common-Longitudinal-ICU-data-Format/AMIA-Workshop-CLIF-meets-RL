# config/

Site-specific configuration for running the workshop notebooks against your
own CLIF data.

## Setup

```bash
cp config_template.json config.json
```

Then edit `config.json` to point at your local CLIF parquet directory:

```json
{
  "site_name": "your_site_name",
  "tables_path": "/absolute/path/to/clif/parquet/",
  "file_type": "parquet",
  "timezone": "US/Eastern",
  "use_demo": false
}
```

| Field | Meaning |
|---|---|
| `site_name` | A label used in printed/logged output. Free text. |
| `tables_path` | Absolute path to a directory containing `clif_*.parquet` files. |
| `file_type` | `parquet` (recommended) or `csv`. |
| `timezone` | IANA timezone of the source data, e.g. `US/Eastern`, `America/Chicago`, `UTC`. |
| `use_demo` | If `true`, ignore `tables_path` and force the bundled `clifpy` demo dataset. Useful when you want to share the same repo state but quickly switch back to demo data. |

## Demo mode

If `config.json` does **not** exist (the default state of a fresh clone, and
the default on Google Colab), the notebooks fall back to the CLIF demo dataset
that ships inside `clifpy`. No setup is required to run them on demo data.

## Security

`config.json` is in `.gitignore`. Do not commit it. It may reveal the file
layout of a site or PHI-bearing directory.
