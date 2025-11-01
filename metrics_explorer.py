# metrics_explorer.py
# CSV explorer with multi-select Project/Run, metric sorting, Unnamed-column removal,
# HARD-CODED metric-name → short-label mapping, and baseline vs. comparison diff highlighting.
# - Fixes Streamlit deprecation: use width='stretch' instead of use_container_width.
# - None/NaN show as empty text and get a white background.
# - Diff column: green when > 0, red otherwise.

import re
from typing import Dict, List
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Metrics Explorer", layout="wide")
st.title("📊 Metrics Explorer")

# ───────────────────────────────────────────────────────────────────────────────
# 1) HARD-CODE YOUR METRIC NAME MAP HERE  (full name -> short label)
# ───────────────────────────────────────────────────────────────────────────────
METRIC_NAME_MAP: Dict[str, str] = {
    # "your full metric name here": "short",
    "metrics/mmlu/0-shot/InContextLearningMultipleChoiceAccuracy": "mmlu0",
    "metrics/mmlu/5-shot/InContextLearningMultipleChoiceAccuracy": "mmlu5",
    "metrics/mmlu-pro/0-shot/InContextLearningMultipleChoiceAccuracy": "mmlu-pro0",
    "metrics/mmlu-pro/5-shot/InContextLearningMultipleChoiceAccuracy": "mmlu-pro5",
    "math/MATH": "MATH",
    "math/MATH_ru": "MATH_ru",
    "math/gsm8k": "gsm8k",
    "math/gsm8k_ru": "gsm8k_ru",
    "code/humaneval/base_pass@1": "humaneval_base@1",
    "code/humaneval/plus_pass@1": "humaneval_plus@1",
    "code/mbpp/base_pass@1": "mbpp_base@1",
    "code/mbpp/plus_pass@1": "mbpp_plus@1",
    "code/lcb/pass@1": "lcb@1",
    "code/lcb/pass@10": "lcb@10",
}

# Optional: auto-shorten any unmapped metric names so more fit on screen
AUTO_SHORTEN_UNMAPPED = True
AUTO_MAX_LEN = 14  # characters (adjust in the sidebar UI)

# Which columns are identifiers (we don't shorten these unless you map them)
IDENTIFIER_HINTS = {"project", "project_id", "proj", "workspace", "team",
                    "run", "run_id", "run_name", "experiment", "exp", "trial", "trial_id"}

# ───────────────────────────────────────────────────────────────────────────────
# Load CSV (upload OR --csv OR default "<script_dir>/data.csv")
# Run: streamlit run metrics_explorer.py -- --csv /path/to/file.csv
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("1) Load CSV")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--csv", type=str, default="metrics_table.csv", help="Path to CSV to load by default")
cli_args, _ = parser.parse_known_args()

uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_df_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop unnamed/index-like columns
    drop_cols = [c for c in df.columns if re.match(r"^Unnamed(:.*)?$", str(c))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

@st.cache_data(show_spinner=False)
def load_df_from_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    drop_cols = [c for c in df.columns if re.match(r"^Unnamed(:.*)?$", str(c))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

df = None
data_source = ""

if uploaded is not None:
    df = load_df_from_uploaded(uploaded)
    data_source = "uploaded file"
else:
    script_dir = Path(__file__).resolve().parent
    if cli_args.csv is not None:
        csv_path = Path(cli_args.csv).expanduser().resolve()
        if not csv_path.exists():
            st.error(f"--csv path not found: {csv_path}")
            st.stop()
        df = load_df_from_path(csv_path)
        data_source = f"--csv ({csv_path})"
    else:
        default_csv = script_dir / "data.csv"  # change name if desired
        if default_csv.exists():
            df = load_df_from_path(default_csv)
            data_source = f"default next to script ({default_csv})"

if df is None or df.empty:
    st.info(
        "No CSV uploaded and no default file found.\n\n"
        "Options:\n"
        "• Upload a CSV in the sidebar\n"
        "• Run with a path:  `streamlit run metrics_explorer.py -- --csv /path/to/file.csv`\n"
        "• Put a `data.csv` next to the script"
    )
    st.stop()

st.sidebar.success(f"Loaded data from: {data_source}")

all_cols: List[str] = list(df.columns)

# ───────────────────────────────────────────────────────────────────────────────
# Detect Project/Run columns (you can override in the sidebar)
# ───────────────────────────────────────────────────────────────────────────────
def detect_col(candidates, df_cols):
    df_lower = {c.lower(): c for c in df_cols}
    for name in candidates:
        if name.lower() in df_lower:
            return df_lower[name.lower()]
    return None

project_guess = detect_col(["project", "project_id", "proj", "workspace", "team"], all_cols)
run_guess = detect_col(["run", "run_id", "run_name", "experiment", "exp", "trial", "trial_id"], all_cols)

# Numeric metric candidates (exclude obvious index/time columns)
exclude_keywords = {"step", "epoch", "iteration", "iter", "time", "timestamp", "date", "id"}
numeric_candidates = [
    c for c in all_cols
    if pd.api.types.is_numeric_dtype(df[c])
    and not any(k in c.lower() for k in exclude_keywords)
]

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ───────────────────────────────────────────────────────────────────────────────
project_col = project_guess or ("project" if "project" in all_cols else None)
run_col = run_guess or ("run_name" if "run_name" in all_cols else None)

if project_col is None:
    st.error("Could not detect the project column. Ensure your CSV includes a 'project' column.")
    st.stop()

if run_col is None:
    st.error("Could not detect the run column. Ensure your CSV includes a 'run_name' column.")
    st.stop()

st.sidebar.header("2) Metrics")

metric_options = numeric_candidates if numeric_candidates else all_cols
metric_col = st.sidebar.selectbox("Metric (for sorting)", options=metric_options, index=0)
order = st.sidebar.selectbox("Order", options=["Descending", "Ascending"], index=0)

st.sidebar.header("3) Filters")

def choices_for(source_df: pd.DataFrame, col: str) -> List[str]:
    vals = source_df[col].dropna().astype(str).unique().tolist()
    vals.sort()
    return vals

projects_selected = st.sidebar.multiselect("Projects (empty = all)", options=choices_for(df, project_col))

project_filtered_df = df
if projects_selected:
    project_filtered_df = project_filtered_df[project_filtered_df[project_col].astype(str).isin(projects_selected)]

run_choices = choices_for(project_filtered_df, run_col)
if not run_choices:
    st.sidebar.warning("No runs available for the selected project filters.")
    st.stop()

baseline_run = st.sidebar.selectbox("Baseline run", options=run_choices, index=0)

comparison_options = [r for r in run_choices if r != baseline_run]
comparison_default = comparison_options[:1] if comparison_options else []
comparison_runs = st.sidebar.multiselect(
    "Comparison runs",
    options=comparison_options,
    default=comparison_default,
)

baseline_run = str(baseline_run)
comparison_runs = [str(r) for r in comparison_runs]

search_term = st.sidebar.text_input("Search (filters all columns)", value="")

st.sidebar.header("4) Display")
AUTO_MAX_LEN = st.sidebar.slider("Auto-shorten length (for unmapped)", 6, 40, AUTO_MAX_LEN)

# ───────────────────────────────────────────────────────────────────────────────
# Build short labels
# ───────────────────────────────────────────────────────────────────────────────
def is_identifier(col: str) -> bool:
    return col.lower() in IDENTIFIER_HINTS or col in {project_col, run_col}

def shorten_name(name: str, max_len: int) -> str:
    n = re.sub(r"\s+", " ", str(name)).strip()
    n = re.sub(r"\b(metric|value|score|val)\b", "", n, flags=re.I).strip()
    if len(n) <= max_len:
        return n
    keep = max_len - 1
    head = max(3, int(keep * 0.6))
    tail = keep - head
    return f"{n[:head]}…{n[-tail:]}" if tail > 0 else f"{n[:keep]}…"

label_map: Dict[str, str] = {}
for col in all_cols:
    if is_identifier(col):
        label_map[col] = col
    else:
        label_map[col] = METRIC_NAME_MAP.get(col, shorten_name(col, AUTO_MAX_LEN) if AUTO_SHORTEN_UNMAPPED else col)

# ───────────────────────────────────────────────────────────────────────────────
# Filtering
# ───────────────────────────────────────────────────────────────────────────────
work = df.copy()

if projects_selected:
    work = work[work[project_col].astype(str).isin(projects_selected)]

selected_runs = [baseline_run] + comparison_runs if baseline_run else comparison_runs
if selected_runs:
    work = work[work[run_col].astype(str).isin(selected_runs)]

if search_term.strip():
    q = search_term.lower().strip()
    mask = work.apply(lambda r: any(q in str(v).lower() for v in r), axis=1)
    work = work[mask]

# ───────────────────────────────────────────────────────────────────────────────
# Sorting by metric
# ───────────────────────────────────────────────────────────────────────────────
ascending = (order == "Ascending")
metric_series = pd.to_numeric(work[metric_col], errors="coerce")
if metric_series.notna().mean() >= 0.5:
    work = work.assign(_metric_sort_key=metric_series).sort_values(
        "_metric_sort_key", ascending=ascending, kind="mergesort"
    ).drop(columns=["_metric_sort_key"])
else:
    work = work.sort_values(metric_col, ascending=ascending, kind="mergesort")


# ───────────────────────────────────────────────────────────────────────────────
# Prepare transposed display DataFrame
# ───────────────────────────────────────────────────────────────────────────────
if work.empty:
    st.warning("No rows match the current filters/search.")
    st.stop()

metric_cols = [col for col in work.columns if col not in {project_col, run_col}]
if not metric_cols:
    st.warning("No metric columns available to display.")
    st.stop()

comparison_runs_sorted = comparison_runs
if comparison_runs:
    sorted_run_order = (
        work.sort_values(metric_col, ascending=ascending, kind="mergesort")[run_col]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    comparison_runs_sorted = [r for r in sorted_run_order if r in comparison_runs]
    if len(comparison_runs_sorted) < len(comparison_runs):
        remaining = [r for r in comparison_runs if r not in comparison_runs_sorted]
        comparison_runs_sorted.extend(remaining)

runs_display_order = [baseline_run] + comparison_runs_sorted
work_runs = work[run_col].astype(str)
runs_in_work = set(work_runs)
missing_runs = [r for r in runs_display_order if r not in runs_in_work]

if missing_runs:
    st.warning(
        "The following selected runs are not present after filtering/search: "
        + ", ".join(missing_runs)
    )

rows = []
for metric in metric_cols:
    metric_label = label_map[metric]
    row_values = {"Metric": metric_label}
    for run in runs_display_order:
        mask = work_runs == run
        series = work.loc[mask, metric]
        if not series.empty:
            value = pd.to_numeric(series, errors="coerce").iloc[0]
        else:
            value = np.nan
        row_values[run] = value
    rows.append(row_values)

display_df = pd.DataFrame(rows)

diff_column_name = None
if comparison_runs_sorted and len(comparison_runs_sorted) == 1:
    comp_run = comparison_runs_sorted[0]
    diff_column_name = "Diff"
    display_df[diff_column_name] = display_df[comp_run] - display_df[baseline_run]

value_columns = [col for col in runs_display_order if col in display_df.columns]
if diff_column_name:
    value_columns.append(diff_column_name)

display_df = display_df[["Metric"] + value_columns]

numeric_display_cols = [col for col in value_columns if pd.api.types.is_numeric_dtype(display_df[col])]

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.subheader("Results")
with right:
    st.metric("Rows", value=len(display_df))

comparison_text = (
    f"Comparisons: **{', '.join(comparison_runs)}**."
    if comparison_runs
    else "Comparisons: **None**."
)
st.caption(
    f"Sorted by **{metric_col}** ({'ascending' if ascending else 'descending'}). "
    f"Projects: **{', '.join(projects_selected) if projects_selected else 'All'}**, "
    f"Baseline: **{baseline_run}**, {comparison_text}"
)

# ───────────────────────────────────────────────────────────────────────────────
# Styling:
# - Format all numeric cells to 5 decimals.
# - Highlight diff column green (>0) or red (<=0).
# - Show None/NaN as empty text and force white background on those cells.
# - Auto-size column min-width so content fits; prevent wrapping.
# ───────────────────────────────────────────────────────────────────────────────
styler = display_df.style

# Render None/NaN as empty strings for readability
styler = styler.format(na_rep="")

# Format numeric columns to 5 decimals
if numeric_display_cols:
    five_dec_formatter = {col: (lambda v: "" if pd.isna(v) else f"{float(v):.5f}") for col in numeric_display_cols}
    styler = styler.format(formatter=five_dec_formatter, na_rep="")

if diff_column_name:
    def color_diff(val):
        if pd.isna(val) or val == "":
            return ""
        return "color: green" if val > 0 else "color: red"
    styler = styler.applymap(color_diff, subset=[diff_column_name])

# White background for None/NaN in all columns — apply last so it overrides prior styling
def white_na(s: pd.Series):
    return ['background-color: white' if pd.isna(v) or v == "" else '' for v in s]

styler = styler.apply(white_na, axis=0)

# ---- Auto column widths so values fit (no clipping) ----
# Compute the max content width (in characters) per displayed column.
width_map: Dict[str, int] = {}
for col_label in display_df.columns:
    series = display_df[col_label]
    if col_label in numeric_display_cols:
        # Use the 5-decimal formatted text for sizing
        texts = series.map(lambda v: "" if pd.isna(v) else f"{float(v):.5f}")
    else:
        texts = series.astype(str).fillna("")
    max_len = max([len(col_label)] + [len(t) for t in texts])
    # Add a little padding; cap very tiny widths
    width_map[col_label] = max(6, max_len + 2)

# Apply min-width and prevent wrapping per column
for col_label, ch in width_map.items():
    styler = styler.set_properties(
        subset=[col_label],
        **{
            "min-width": f"{ch}ch",
            "white-space": "nowrap",
        }
    )

# Render styled table (Streamlit’s newer API: width='stretch')
st.dataframe(
    styler,
    width='stretch',
    hide_index=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# Download (keeps original headers)
# ───────────────────────────────────────────────────────────────────────────────
def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download current view (CSV, original headers)",
    data=to_csv_bytes(work),
    file_name="metrics_explorer_filtered.csv",
    mime="text/csv",
    width='stretch',
)
