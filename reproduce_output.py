"""Reproduce the descriptive statistics from ``spss_output_A2.spv`` in pandas.

The SPSS viewer file contains two kinds of analysis on ``dataset_2A.sav``:

* **Frequencies** - statistics, frequency tables, a bar chart (GENDER) and
  histograms (HEIGHT, WEIGHT, ID);
* **Means** - HEIGHT (and WEIGHT) broken down by GENDER.

This script recomputes the same quantities and saves a summary figure.

    python reproduce_output.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
GENDER = {0: "male", 1: "female"}


def spss_skewness(s: pd.Series) -> tuple[float, float]:
    """Return (skewness, std. error of skewness) using SPSS's definitions."""
    s = s.dropna()
    n = len(s)
    se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    return s.skew(), se  # pandas .skew() is the bias-adjusted G1 that SPSS reports


def main() -> None:
    df, _ = pyreadstat.read_sav(HERE / "data/dataset_2A.sav")
    df["GENDER_LABEL"] = df["GENDER"].map(GENDER)

    # ---- Frequencies > Statistics ----------------------------------------------
    rows = {}
    for col in ("HEIGHT", "WEIGHT"):
        s = df[col]
        skew, se = spss_skewness(s)
        rows[col] = {
            "N valid": s.count(),
            "N missing": s.isna().sum(),
            "Mean": s.mean(),
            "Mode": s.mode().min(),
            "Std. deviation": s.std(),
            "Skewness": skew,
            "Std. error of skewness": se,
            "Minimum": s.min(),
            "Maximum": s.max(),
            "Sum": s.sum(),
        }
    print("Frequencies - Statistics\n")
    print(pd.DataFrame(rows).round(3).to_string())

    # ---- Frequencies > Frequency Table (GENDER) --------------------------------
    counts = df["GENDER_LABEL"].value_counts().reindex(["male", "female"])
    table = pd.DataFrame(
        {
            "Frequency": counts,
            "Percent": counts / len(df) * 100,
            "Valid percent": counts / counts.sum() * 100,
        }
    )
    n_missing = df["GENDER"].isna().sum()
    print(f"\nFrequencies - GENDER (N = {len(df)}, missing = {n_missing})\n")
    print(table.round(1).to_string())

    # ---- Means ------------------------------------------------------------------
    means = df.groupby("GENDER_LABEL")[["HEIGHT", "WEIGHT"]].agg(
        ["count", "mean", "std"]
    )
    print("\nMeans - by GENDER\n")
    print(means.round(2).to_string())

    # ---- Figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].bar(counts.index, counts.values, color=["#1f4e79", "#c0392b"])
    axes[0].set(title="GENDER (frequency)", ylabel="count")
    axes[1].hist(df["HEIGHT"].dropna(), bins=6, color="#9bb7d4", edgecolor="white")
    axes[1].set(title="HEIGHT (inches)", ylabel="count")
    axes[2].hist(df["WEIGHT"].dropna(), bins=6, color="#9bd4b7", edgecolor="white")
    axes[2].set(title="WEIGHT (as labelled: kg)", ylabel="count")
    means["HEIGHT"]["mean"].plot.bar(
        ax=axes[3], color=["#c0392b", "#1f4e79"], rot=0, yerr=means["HEIGHT"]["std"]
    )
    axes[3].set(title="Mean HEIGHT by GENDER (± sd)", xlabel="", ylabel="inches")
    for ax in axes[:3]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = HERE / "figures/descriptives.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
