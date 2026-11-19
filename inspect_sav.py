"""Print the data dictionary and a preview of one or more SPSS ``.sav`` files.

Usage::

    python inspect_sav.py data/dataset_2A.sav
    python inspect_sav.py --markdown data/dataset_2A.sav   # GitHub-ready table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyreadstat


def dictionary(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(data, dictionary)`` for an SPSS file.

    The dictionary has one row per variable: label, SPSS measurement level,
    number of missing values, and any value labels (e.g. ``0=male, 1=female``).
    """
    df, meta = pyreadstat.read_sav(path)
    rows = []
    for name in meta.column_names:
        labels = meta.variable_value_labels.get(name, {})
        rows.append(
            {
                "variable": name,
                "label": meta.column_names_to_labels.get(name) or "",
                "measure": meta.variable_measure.get(name, ""),
                "missing": int(df[name].isna().sum()),
                "value labels": ", ".join(f"{int(k)}={v}" for k, v in labels.items()),
            }
        )
    return df, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit the dictionary as a Markdown table",
    )
    args = parser.parse_args()

    for path in args.files:
        df, dic = dictionary(path)
        print(f"\n## {path.name}  ({len(df)} rows x {df.shape[1]} columns)\n")
        print(
            dic.to_markdown(index=False)
            if args.markdown
            else dic.to_string(index=False)
        )
        if not args.markdown:
            print("\nPreview:\n" + df.head().to_string())


if __name__ == "__main__":
    main()
