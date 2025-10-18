#!/usr/bin/env python3
# analyze_coffee_survey.py
#
# Basic EDA for the coffee survey described in the prompt.
# Usage: python analyze_coffee_survey.py input.csv

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Helpers

def read_csv_safely(path: str) -> pd.DataFrame:
    # Many survey exports are UTF-8, sometimes UTF-16/latin-1. Try a few encodings.
    encodings = ["utf-8-sig", "utf-8", "utf-16", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings. Last error: {last_err}")

def coerce_numeric(series: pd.Series) -> pd.Series:
    """Turn things like '$45', ' 30 ', '30.0', '30 dollars' into floats."""
    if series is None:
        return series
    s = series.astype(str).str.replace(r"[,$]", "", regex=True)
    s = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def clean_check(series: pd.Series) -> pd.Series:
    """
    Interpret multi-select checkbox style columns.
    Treat non-empty / truthy values (Yes/TRUE/1/Checked) as True.
    """
    if series is None:
        return series
    s = series.copy()
    s = s.astype(str).str.strip().str.lower()
    truthy = {"yes", "y", "true", "t", "1", "checked", "x"}
    falsy  = {"no", "n", "false", "f", "0", ""}
    out = []
    for v in s:
        if v in truthy:
            out.append(True)
        elif v in falsy:
            out.append(False)
        else:
            # If it's some text (e.g., the option label), count as True
            out.append(True)
    return pd.Series(out, index=series.index)

def group_checkbox_columns(df: pd.DataFrame) -> dict:
    """
    Group columns that share a stem like:
    'How do you brew coffee at home?' and
    'How do you brew coffee at home? (French press)'
    Returns: { group_stem: [list of subcolumns] }
    """
    groups = defaultdict(list)
    for col in df.columns:
        m = re.match(r"^(.*?\?)\s*\(", col)  # e.g., "Question? (Option)"
        if m:
            stem = m.group(1).strip()
            groups[stem].append(col)
    # Also group non-question stems that use parentheses frequently (e.g., Where do you... etc.)
    # If a column has "(...)" and shares a long prefix before ' (', group by that prefix.
    for col in df.columns:
        if col in [c for cols in groups.values() for c in cols]:
            continue
        if " (" in col and ")" in col:
            stem = col.split(" (")[0].strip()
            # If there are at least 2 columns with this stem, consider it a group
            matches = [c for c in df.columns if c.startswith(stem + " (")]
            if len(matches) >= 2:
                groups[stem].extend(matches)
    # Deduplicate & sort
    for k in list(groups.keys()):
        groups[k] = sorted(set(groups[k]), key=lambda x: df.columns.get_loc(x))
    return dict(groups)

def top_value_counts(series: pd.Series, n=10) -> pd.Series:
    return series.dropna().astype(str).str.strip().replace({"": np.nan}).dropna().value_counts().head(n)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_series_counts(s: pd.Series, title: str, outpath: str):
    if s.empty:
        return
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---------- Analysis steps

def analyze(df: pd.DataFrame, outdir: str) -> str:
    ensure_dir(outdir)
    lines = []

    # --- Basic shape
    n_rows, n_cols = df.shape
    lines.append(f"Rows: {n_rows:,}")
    lines.append(f"Columns: {n_cols:,}")

    # --- Missingness overview
    miss = df.isna().mean().sort_values(ascending=False)
    miss_path = os.path.join(outdir, "missingness.csv")
    miss.to_csv(miss_path, header=["fraction_missing"])
    lines.append(f"Saved missingness by column -> {miss_path}")

    # --- Age distribution (if present)
    age_col = "What is your age?"
    if age_col in df.columns:
        age = coerce_numeric(df[age_col])
        age_desc = age.describe(percentiles=[0.25, 0.5, 0.75]).to_string()
        lines.append("\nAge (summary):")
        lines.append(age_desc)
        # Binned
        bins = [0, 17, 24, 34, 44, 54, 64, 74, 150]
        labels = ["<=17","18-24","25-34","35-44","45-54","55-64","65-74","75+"]
        age_bins = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
        age_counts = age_bins.value_counts().sort_index()
        age_counts_path = os.path.join(outdir, "age_bins.csv")
        age_counts.to_csv(age_counts_path, header=["count"])
        plot_series_counts(age_counts, "Age Distribution", os.path.join(outdir, "age_distribution.png"))
        lines.append(f"Saved age bins -> {age_counts_path}")

    # --- Cups per day
    cups_col = "How many cups of coffee do you typically drink per day?"
    if cups_col in df.columns:
        cups = coerce_numeric(df[cups_col])
        lines.append("\nCups per day (summary):")
        lines.append(cups.describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        cups_counts = cups.round().value_counts().sort_index()
        plot_series_counts(cups_counts, "Cups of Coffee per Day (rounded)", os.path.join(outdir, "cups_per_day.png"))

    # --- Group checkbox-style questions
    groups = group_checkbox_columns(df)
    group_dir = os.path.join(outdir, "checkbox_groups")
    ensure_dir(group_dir)
    if groups:
        lines.append("\nMulti-select groups found:")
        for stem, cols in groups.items():
            lines.append(f"- {stem} ({len(cols)} options)")
            # Count each option as number of True/checked responses
            counts = {}
            for c in cols:
                checked = clean_check(df[c])
                counts[c.replace(stem, "").strip() or c] = checked.sum(skipna=True)
            counts_s = pd.Series(counts).sort_values(ascending=False)
            csv_path = os.path.join(group_dir, f"{re.sub(r'[^a-zA-Z0-9_]+','_', stem)[:60]}.csv")
            counts_s.to_csv(csv_path, header=["count"])
            plot_series_counts(counts_s, stem, os.path.join(group_dir, f"{re.sub(r'[^a-zA-Z0-9_]+','_', stem)[:60]}.png"))

    # --- Favorite coffee drink (free text)
    fav_col = "Please specify what your favorite coffee drink is"
    if fav_col in df.columns:
        fav_top = top_value_counts(df[fav_col], n=15)
        fav_path = os.path.join(outdir, "favorite_drinks_top.csv")
        fav_top.to_csv(fav_path, header=["count"])
        lines.append("\nTop favorite coffee drinks (free text, top 15):")
        lines.append(fav_top.to_string())
        plot_series_counts(fav_top, "Favorite Coffee Drinks (Top 15)", os.path.join(outdir, "favorite_drinks_top.png"))

    # --- Value perception & yes/no style items (simple counts)
    yn_candidates = [
        "Do you usually add anything to your coffee?",
        "Do you like the taste of coffee?",
        "Do you know where your coffee comes from?",
        "Do you feel like you’re getting good value for your money when you buy coffee at a cafe?",
        "Do you feel like you’re getting good value for your money with regards to your coffee equipment?",
        "Do you work from home or in person?",
    ]
    for col in yn_candidates:
        if col in df.columns:
            vc = df[col].dropna().astype(str).str.strip().value_counts()
            p = os.path.join(outdir, f"{re.sub(r'[^a-zA-Z0-9_]+','_', col)[:60]}_counts.csv")
            vc.to_csv(p, header=["count"])
            plot_series_counts(vc, col, os.path.join(outdir, f"{re.sub(r'[^a-zA-Z0-9_]+','_', col)[:60]}.png"))
            lines.append(f"\n{col} (counts):\n{vc.to_string()}")

    # --- Spending (monthly coffee, equipment)
    spend_cols = [
        "In total, much money do you typically spend on coffee in a month?",
        "Approximately how much have you spent on coffee equipment in the past 5 years?",
        "What is the most you've ever paid for a cup of coffee?",
        "What is the most you'd ever be willing to pay for a cup of coffee?",
    ]
    for col in spend_cols:
        if col in df.columns:
            nums = coerce_numeric(df[col])
            desc = nums.describe(percentiles=[0.25, 0.5, 0.75]).to_string()
            lines.append(f"\n{col} (summary):\n{desc}")

    # --- Coffee A-D ratings (means if present)
    for letter in ["A", "B", "C", "D"]:
        for facet in ["Bitterness", "Acidity", "Personal Preference"]:
            col = f"Coffee {letter} - {facet}"
            if col in df.columns:
                nums = coerce_numeric(df[col])
                lines.append(f"Coffee {letter} - {facet}: mean={nums.mean():.2f} (n={nums.notna().sum()})")

    # --- Preference between coffees
    pref_cols = [
        "Between Coffee A, Coffee B, and Coffee C which did you prefer?",
        "Between Coffee A and Coffee D, which did you prefer?",
        "Lastly, what was your favorite overall coffee?",
    ]
    for col in pref_cols:
        if col in df.columns:
            vc = df[col].dropna().astype(str).str.strip().value_counts()
            lines.append(f"\n{col} (counts):\n{vc.to_string()}")

    # --- Demographics quick counts
    demo_cols = [
        "Gender", "Education Level", "Ethnicity/Race",
        "Employment Status", "Number of Children", "Political Affiliation"
    ]
    for col in demo_cols:
        if col in df.columns:
            vc = df[col].dropna().astype(str).str.strip().value_counts()
            p = os.path.join(outdir, f"demo_{re.sub(r'[^a-zA-Z0-9_]+','_', col)[:40]}.csv")
            vc.to_csv(p, header=["count"])

    # --- Save a text summary
    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Basic analysis for the coffee survey CSV.")
    parser.add_argument("csv_path", help="Path to the survey CSV file")
    parser.add_argument("--outdir", default="coffee_report", help="Directory to write outputs")
    args = parser.parse_args()

    df = read_csv_safely(args.csv_path)
    summary_path = analyze(df, args.outdir)
    print(f"\nDone. Summary written to: {summary_path}")
    print(f"Artifacts written to folder: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
