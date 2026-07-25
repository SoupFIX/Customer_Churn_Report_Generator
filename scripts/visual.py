"""
generic_eda_visualizer.py
--------------------------
A dataset-agnostic version of the churn-analysis chart script.

Instead of hardcoding column names like 'contract', 'internetservice',
'churn', etc., this script INSPECTS whatever file you give it and
auto-detects:
    - a binary/target column (e.g. churn, default, purchased, converted...)
    - categorical columns (low unique-value text/object columns)
    - numeric columns

...then generates the same *style* of charts (target distribution,
target-rate-by-category, numeric distributions split by target,
correlation heatmap, top category combo) for WHATEVER dataset you feed it.

USAGE
-----
    python generic_eda_visualizer.py path/to/your_file.csv
    python generic_eda_visualizer.py data.xlsx --target Churn
    python generic_eda_visualizer.py data.csv --target Purchased --output charts_out

If you don't pass --target, the script tries to auto-detect a sensible
binary column (something with exactly 2 unique values, ideally named
something like churn/target/label/purchased/default/converted/outcome).
If it can't find one confidently, it will list candidate columns and
ask you to re-run with --target explicitly.
"""

import argparse
import string
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
})

BLUE = '#4C72B0'
ORANGE = '#DD8452'
GREY = '#B0B0B0'

# Column-name hints used only for auto-detecting a likely target column.
TARGET_NAME_HINTS = [
    'churn', 'target', 'label', 'purchased', 'default', 'converted',
    'outcome', 'result', 'response', 'attrition', 'exited', 'is_fraud',
    'fraud', 'clicked', 'subscribed', 'flag'
]

# Values that count as "positive" when a target column is text-based.
POSITIVE_VALUES = {'yes', 'true', '1', 'churned', 'purchased', 'converted',
                    'default', 'fraud', 'subscribed', 'exited', 'attrited'}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_file(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    elif path.suffix.lower() == '.json':
        df = pd.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {path.suffix}. "
            f"Supported formats are: .csv, .xlsx, .xls, .json"
        )

    # Normalize column names once, up front, so the rest of the script
    # never has to guess about capitalization/whitespace again.
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df


def get_latest_file(folder_path, pattern="*"):
    folder = Path(folder_path)
    files = list(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {folder_path}")
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest


# ---------------------------------------------------------------------------
# Column-type detection
# ---------------------------------------------------------------------------
def detect_target_column(df: pd.DataFrame, user_choice: str | None) -> str:
    if user_choice:
        col = user_choice.strip().lower().replace(' ', '_')
        if col not in df.columns:
            raise ValueError(
                f"--target '{user_choice}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return col

    # 1. Prefer a column whose name matches a common target keyword
    #    AND has exactly 2 unique non-null values.
    for col in df.columns:
        if any(hint in col for hint in TARGET_NAME_HINTS):
            if df[col].nunique(dropna=True) == 2:
                return col

    # 2. Otherwise, fall back to any binary (2-unique-value) column at all.
    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if len(binary_cols) == 1:
        return binary_cols[0]

    # 3. Can't decide confidently -> stop and ask the user.
    raise ValueError(
        "Could not auto-detect a target/outcome column.\n"
        f"Candidate binary columns found: {binary_cols}\n"
        "Re-run with --target <column_name> to specify which one to use."
    )


def make_target_flag(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Turn any binary column (Yes/No, True/False, 1/0, custom text) into 0/1."""
    series = df[target_col]

    if pd.api.types.is_numeric_dtype(series):
        uniq = sorted(series.dropna().unique())
        if set(uniq) <= {0, 1}:
            return series.astype(int)
        # Numeric but not already 0/1 -> treat the larger value as positive
        return (series == uniq[-1]).astype(int)

    # Text/categorical target
    return series.astype(str).str.strip().str.lower().isin(POSITIVE_VALUES).astype(int)


def detect_categorical_columns(df: pd.DataFrame, target_col: str, max_unique: int = 10) -> list:
    excluded = {target_col, 'churn_flag', 'target_flag'}
    cats = []
    for col in df.columns:
        if col in excluded:
            continue
        is_text_like = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
        if is_text_like:
            if 1 < df[col].nunique(dropna=True) <= max_unique:
                cats.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) <= max_unique:
            cats.append(col)
    return cats


def detect_numeric_columns(df: pd.DataFrame, target_col: str) -> list:
    excluded = {target_col, 'churn_flag', 'target_flag'}
    id_like_hints = ('id', 'index', 'key', 'code', 'number', 'zip', 'phone')
    nums = []
    for col in df.columns:
        if col in excluded:
            continue
        if any(hint in col for hint in id_like_hints):
            continue  # skip ID-like columns (e.g. customerid) — not useful for distributions
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 10:
            nums.append(col)
    return nums


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def chart_target_distribution(df, target_col, flag_col, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[flag_col].value_counts().sort_index()
    labels = ['Negative (0)', 'Positive (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, values, color=[GREY, ORANGE])
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + total * 0.01,
                f'{h:,}\n({h / total * 100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    pos_rate = values[1] / total * 100 if total else 0
    ax.set_title(f"'{target_col}' Positive Rate: {pos_rate:.1f}%", fontweight='bold')
    ax.set_ylabel('Number of Rows')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / '01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def chart_rate_by_category(df, cat_col, flag_col, out_dir, idx):
    rate = df.groupby(cat_col)[flag_col].mean().mul(100).sort_values()
    if rate.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(rate))))
    colors = [BLUE if v < rate.mean() else ORANGE for v in rate.values]
    bars = ax.barh(rate.index.astype(str), rate.values, color=colors)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f'{w:.1f}%', va='center', fontsize=10)
    ax.set_title(f'Positive Rate by {cat_col}', fontweight='bold')
    ax.set_xlabel('Positive Rate (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f'{idx:02d}_rate_by_{cat_col}.png', dpi=150, bbox_inches='tight')
    plt.close()


def chart_numeric_distribution(df, num_col, flag_col, out_dir, idx):
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = df.loc[df[flag_col] == 1, num_col].dropna()
    neg = df.loc[df[flag_col] == 0, num_col].dropna()
    if pos.empty or neg.empty:
        plt.close()
        return
    ax.hist(neg, bins=30, alpha=0.6, label='Negative', color=BLUE, density=True)
    ax.hist(pos, bins=30, alpha=0.6, label='Positive', color=ORANGE, density=True)
    ax.axvline(neg.median(), color=BLUE, linestyle='--', linewidth=1.5)
    ax.axvline(pos.median(), color=ORANGE, linestyle='--', linewidth=1.5)
    ax.set_title(f'{num_col}: Distribution by Target (median {neg.median():.1f} vs {pos.median():.1f})',
                fontweight='bold')
    ax.set_xlabel(num_col)
    ax.set_ylabel('Density')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f'{idx:02d}_distribution_{num_col}.png', dpi=150, bbox_inches='tight')
    plt.close()


def chart_numeric_boxplot(df, num_col, flag_col, out_dir, idx):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = df[[flag_col, num_col]].dropna().copy()
    plot_df[flag_col] = plot_df[flag_col].map({0: 'Negative', 1: 'Positive'})
    sns.boxplot(data=plot_df, x=flag_col, y=num_col, order=['Negative', 'Positive'],
                hue=flag_col, palette=[BLUE, ORANGE], legend=False, ax=ax)
    ax.set_title(f'{num_col} by Target Group', fontweight='bold')
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f'{idx:02d}_boxplot_{num_col}.png', dpi=150, bbox_inches='tight')
    plt.close()


def chart_correlation_heatmap(df, numeric_cols, flag_col, out_dir, idx):
    cols = [c for c in numeric_cols if c in df.columns] + [flag_col]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols)), max(5, 0.6 * len(cols))))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation with Target', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f'{idx:02d}_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def chart_top_category_combo(df, cat_cols, flag_col, out_dir, idx):
    if len(cat_cols) < 2:
        return
    c1, c2 = cat_cols[0], cat_cols[1]
    pivot = df.groupby([c1, c2])[flag_col].mean().mul(100).unstack()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind='bar', ax=ax, color=[BLUE, ORANGE, GREY, '#8172B2', '#937860'][:pivot.shape[1]])
    ax.set_title(f'Positive Rate by {c1} x {c2}', fontweight='bold')
    ax.set_ylabel('Positive Rate (%)')
    ax.set_xlabel('')
    ax.legend(title=c2)
    plt.xticks(rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / f'{idx:02d}_combo_{c1}_{c2}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():    
    parser = argparse.ArgumentParser(
        description="Generalized EDA chart generator — works on any CSV/XLSX/XLS/JSON file."
    )

    latest_file = get_latest_file("../cleaned data/")
    parser.add_argument("--target", default=None,
                        help="Name of the binary target/outcome column. "
                        "If omitted, the script tries to auto-detect one.")
    parser.add_argument("--max-category-charts", type=int, default=6,
                        help="Max number of 'rate by category' charts to generate (default 6).")
    parser.add_argument("--max-numeric-charts", type=int, default=6,
                        help="Max number of numeric distribution/boxplot charts to generate (default 6).")
    args = parser.parse_args()

    df = load_file(latest_file)

    target_col = detect_target_column(df, args.target)
    df['target_flag'] = make_target_flag(df, target_col)

    categorical_cols = detect_categorical_columns(df, target_col)
    numeric_cols = detect_numeric_columns(df, target_col)

    print(f"Detected target column : {target_col}")
    print(f"Categorical columns    : {categorical_cols}")
    print(f"Numeric columns        : {numeric_cols}")

    # Output folder
    date_part = datetime.now().strftime("charts_%Y-%m-%d")
    suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
    out_dir = Path(f"../charts/{date_part}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    chart_idx = 1
    chart_target_distribution(df, target_col, 'target_flag', out_dir)
    chart_idx += 1

    for col in categorical_cols[:args.max_category_charts]:
        chart_rate_by_category(df, col, 'target_flag', out_dir, chart_idx)
        chart_idx += 1

    for col in numeric_cols[:args.max_numeric_charts]:
        chart_numeric_distribution(df, col, 'target_flag', out_dir, chart_idx)
        chart_idx += 1
        chart_numeric_boxplot(df, col, 'target_flag', out_dir, chart_idx)
        chart_idx += 1

    chart_correlation_heatmap(df, numeric_cols, 'target_flag', out_dir, chart_idx)
    chart_idx += 1

    chart_top_category_combo(df, categorical_cols, 'target_flag', out_dir, chart_idx)
    chart_idx += 1

    print(f"\nAll charts saved to: {out_dir.resolve()}")
    print("\n--- Key numbers ---")
    print(f"Overall positive rate: {df['target_flag'].mean() * 100:.2f}%")
    for col in categorical_cols[:args.max_category_charts]:
        print(f"\nRate by {col}:")
        print(df.groupby(col)['target_flag'].mean().mul(100).round(2).sort_values())


if __name__ == "__main__":
    main()