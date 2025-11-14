# metrics_explorer.py
# CSV explorer with multi-select Project/Run, metric sorting, Unnamed-column removal,
# HARD-CODED metric-name → short-label mapping, and baseline vs. comparison diff highlighting.
# - Fixes Streamlit deprecation: use width='stretch' instead of use_container_width.
# - None/NaN show as empty text and get a white background.
# - Comparison run values: only color when statistically significant (p < 0.05), with stronger shading for lower p-values.

import io
import re
from typing import Dict, List, Optional
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import streamlit as st
from math import erf, sqrt
import boto3

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

SAMPLE_SIZE_BY_METRIC: Dict[str, int] = {
    "metrics/mmlu/0-shot/InContextLearningMultipleChoiceAccuracy": 14079,
    "metrics/mmlu/5-shot/InContextLearningMultipleChoiceAccuracy": 14079,
    "mmlu0": 14079,
    "mmlu5": 14079,
    "metrics/mmlu-pro/0-shot/InContextLearningMultipleChoiceAccuracy": 12032,
    "metrics/mmlu-pro/5-shot/InContextLearningMultipleChoiceAccuracy": 12032,
    "mmlu-pro0": 12032,
    "mmlu-pro5": 12032,
    "math/MATH": 500,
    "MATH": 500,
    "math/MATH_ru": 500,
    "MATH_ru": 500,
    "math/gsm8k": 1319,
    "gsm8k": 1319,
    "math/gsm8k_ru": 1319,
    "gsm8k_ru": 1319,
    "code/humaneval/base_pass@1": 164,
    "humaneval_base@1": 164,
    "code/humaneval/plus_pass@1": 164,
    "humaneval_plus@1": 164,
    "code/mbpp/base_pass@1": 427,
    "mbpp_base@1": 427,
    "code/mbpp/plus_pass@1": 378,
    "mbpp_plus@1": 378,
    "code/lcb/pass@1": 1055,
    "code/lcb/pass@10": 1055,
    "lcb@1": 1055,
    "lcb@10": 1055,
}

def get_sample_size_for_metric(metric_name: str, metric_label: str) -> float:
    candidates = [
        metric_name,
        metric_label,
        metric_name.lower(),
        metric_label.lower(),
    ]
    for key in candidates:
        if key in SAMPLE_SIZE_BY_METRIC:
            return SAMPLE_SIZE_BY_METRIC[key]
    return np.nan

def normalize_probability(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        val = float(value)
    except (TypeError, ValueError):
        return np.nan
    if val < 0:
        return np.nan
    if 1.0 < val <= 100.0:
        val = val / 100.0
    if val < 0 or val > 1:
        return np.nan
    return val

def two_proportion_p_value(p1: float, p2: float, n: float) -> float:
    if (
        pd.isna(p1) or pd.isna(p2) or pd.isna(n)
        or n <= 0
    ):
        return np.nan
    pooled = (p1 + p2) / 2.0
    variance = pooled * (1.0 - pooled) * (2.0 / n)
    if variance <= 0:
        return np.nan
    z = (p1 - p2) / np.sqrt(variance)
    cdf = 0.5 * (1 + erf(abs(z) / sqrt(2)))
    return max(0.0, min(1.0, 2 * (1 - cdf)))

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
parser.add_argument("--csv", type=str, default=None, help="Path to CSV to load by default")
cli_args, _ = parser.parse_known_args()

uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if re.match(r"^Unnamed(:.*)?$", str(c))]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

@st.cache_data(show_spinner=False)
def load_df_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _drop_unnamed_columns(df)

@st.cache_data(show_spinner=False)
def load_df_from_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _drop_unnamed_columns(df)

@st.cache_data(show_spinner=False)
def load_df_from_s3(
    bucket: str,
    key: str,
    region: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> pd.DataFrame:
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        endpoint_url=endpoint_url,
    )
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    df = pd.read_csv(io.BytesIO(body))
    return _drop_unnamed_columns(df)


def get_s3_default_config() -> Optional[Dict[str, str]]:
    secrets_obj = getattr(st, "secrets", None)
    if secrets_obj is None:
        return None
    section = None
    try:
        section = secrets_obj["s3_default_csv"]
    except Exception:
        section = secrets_obj.get("s3_default_csv") if hasattr(secrets_obj, "get") else None
    if not section:
        return None
    if not isinstance(section, dict):
        try:
            section = dict(section)
        except Exception:
            section = None
    if not section:
        return None
    bucket = section.get("bucket")
    key = section.get("key")
    if not bucket or not key:
        return None
    return section

df = None
data_source = ""
s3_config = get_s3_default_config()

if uploaded is not None:
    df = load_df_from_uploaded(uploaded)
    data_source = "uploaded file"
else:
    script_dir = Path(__file__).resolve().parent
    if cli_args.csv:
        csv_path = Path(cli_args.csv).expanduser().resolve()
        if not csv_path.exists():
            st.error(f"--csv path not found: {csv_path}")
            st.stop()
        df = load_df_from_path(csv_path)
        data_source = f"--csv ({csv_path})"
    elif s3_config is not None:
        try:
            df = load_df_from_s3(
                bucket=s3_config.get("bucket"),
                key=s3_config.get("key"),
                region=s3_config.get("region"),
                aws_access_key_id=s3_config.get("aws_access_key_id"),
                aws_secret_access_key=s3_config.get("aws_secret_access_key"),
                aws_session_token=s3_config.get("aws_session_token"),
                endpoint_url=s3_config.get("endpoint_url"),
            )
            data_source = "default csv"
        except Exception as exc:
            st.error(f"Failed to load default CSV from S3: {exc}")
            st.stop()
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
        "• Put a `data.csv` next to the script\n"
        "• Configure `.streamlit/secrets.toml` with an `[s3_default_csv]` section"
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

project_options = choices_for(df, project_col)
PROJECT_MULTI_KEY = "project_multiselect"
if PROJECT_MULTI_KEY not in st.session_state:
    st.session_state[PROJECT_MULTI_KEY] = []

st.session_state[PROJECT_MULTI_KEY] = [
    p for p in st.session_state[PROJECT_MULTI_KEY] if p in project_options
]

def _select_all_projects():
    st.session_state[PROJECT_MULTI_KEY] = project_options.copy()

st.sidebar.button("Select all projects", key="select_all_projects_btn", on_click=_select_all_projects)

projects_selected = st.sidebar.multiselect(
    "Projects (empty = all)",
    options=project_options,
    key=PROJECT_MULTI_KEY,
)

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
COMPARISON_MULTI_KEY = "comparison_runs_multiselect"
if COMPARISON_MULTI_KEY not in st.session_state:
    st.session_state[COMPARISON_MULTI_KEY] = comparison_default.copy()
else:
    st.session_state[COMPARISON_MULTI_KEY] = [
        r for r in st.session_state[COMPARISON_MULTI_KEY] if r in comparison_options
    ]
    if not st.session_state[COMPARISON_MULTI_KEY] and comparison_default:
        st.session_state[COMPARISON_MULTI_KEY] = comparison_default.copy()

def _select_all_runs():
    st.session_state[COMPARISON_MULTI_KEY] = comparison_options.copy()

st.sidebar.button("Select all runs", key="select_all_runs_btn", on_click=_select_all_runs)

comparison_runs = st.sidebar.multiselect(
    "Comparison runs",
    options=comparison_options,
    key=COMPARISON_MULTI_KEY,
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
    sample_size = get_sample_size_for_metric(metric, metric_label)
    row_values = {"Metric": metric_label, "_sample_size": sample_size}
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

run_value_cols = [col for col in runs_display_order if col in display_df.columns]
if run_value_cols:
    display_df = display_df[~display_df[run_value_cols].isna().all(axis=1)].reset_index(drop=True)

if display_df.empty:
    st.warning("No metrics have values for the selected runs.")
    st.stop()

sample_size_series = (
    display_df["_sample_size"]
    if "_sample_size" in display_df.columns
    else pd.Series(np.nan, index=display_df.index)
)

baseline_for_calcs = display_df[baseline_run] if baseline_run in display_df.columns else None
run_diffs: Dict[str, pd.Series] = {}
run_pvalues: Dict[str, pd.Series] = {}

if baseline_for_calcs is not None:
    for run in comparison_runs_sorted:
        if run not in display_df.columns:
            continue
        comp_series = display_df[run]
        diff_series = comp_series - baseline_for_calcs
        run_diffs[run] = diff_series
        p_values: List[float] = []
        for base_val, comp_val, sample_size in zip(baseline_for_calcs, comp_series, sample_size_series):
            base_prob = normalize_probability(base_val)
            comp_prob = normalize_probability(comp_val)
            p_values.append(two_proportion_p_value(base_prob, comp_prob, sample_size))
        run_pvalues[run] = pd.Series(p_values, index=display_df.index)

diff_column_name = None
comp_run = None
if comparison_runs_sorted and len(comparison_runs_sorted) == 1:
    comp_run = comparison_runs_sorted[0]
    diff_column_name = "Diff"
    if comp_run in run_diffs:
        display_df[diff_column_name] = run_diffs[comp_run]
    else:
        display_df[diff_column_name] = display_df[comp_run] - display_df.get(baseline_run, np.nan)

p_value_column_name = None
if diff_column_name:
    p_value_column_name = "P-value"
    p_series = run_pvalues.get(comp_run)
    if p_series is not None:
        display_df[p_value_column_name] = p_series.values
    else:
        display_df[p_value_column_name] = np.nan

if "_sample_size" in display_df.columns:
    display_df = display_df.drop(columns=["_sample_size"])

run_columns_in_display = [col for col in runs_display_order if col in display_df.columns]
selected_run_count = len(run_columns_in_display)

value_columns = run_columns_in_display.copy()
if diff_column_name:
    value_columns.append(diff_column_name)
if p_value_column_name:
    value_columns.append(p_value_column_name)

display_df = display_df[["Metric"] + value_columns]

numeric_display_cols = [col for col in value_columns if pd.api.types.is_numeric_dtype(display_df[col])]

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.subheader("Results")
with right:
    st.metric("Selected runs", value=selected_run_count)

# ───────────────────────────────────────────────────────────────────────────────
# Styling:
# - Format all numeric cells to 5 decimals.
# - Highlight comparison value columns only when p-value <0.05 (stronger shade for p-value <0.01).
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

highlight_runs = [run for run in comparison_runs_sorted if run in display_df.columns]

if baseline_for_calcs is not None and highlight_runs:
    strong_green = "background-color: #2f855a; color: white"
    light_green = "background-color: #9ae6b4; color: #1c4532"
    strong_red = "background-color: #c53030; color: white"
    light_red = "background-color: #feb2b2; color: #742a2a"

    def make_value_highlighter(diff_series: pd.Series, p_series: Optional[pd.Series]):
        def _highlight(col: pd.Series) -> List[str]:
            styles: List[str] = []
            for idx, comp_val in col.items():
                if pd.isna(comp_val):
                    styles.append("")
                    continue
                diff_val = diff_series.loc[idx] if diff_series is not None else np.nan
                if pd.isna(diff_val) or diff_val == 0:
                    styles.append("")
                    continue
                p_val = p_series.loc[idx] if (p_series is not None and idx in p_series.index) else np.nan
                if pd.isna(p_val) or p_val >= 0.05:
                    styles.append("")
                    continue
                if diff_val > 0:
                    if pd.notna(p_val) and p_val < 0.01:
                        styles.append(strong_green)
                    else:
                        styles.append(light_green)
                else:
                    if pd.notna(p_val) and p_val < 0.01:
                        styles.append(strong_red)
                    else:
                        styles.append(light_red)
            return styles
        return _highlight

    for run in highlight_runs:
        diff_series = run_diffs.get(run)
        if diff_series is None:
            continue
        run_p_series = run_pvalues.get(run)
        styler = styler.apply(make_value_highlighter(diff_series, run_p_series), subset=[run])

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

# Render styled table (Streamlit’s newer API: width='stretch') and lift the height cap so the full table shows
# without Streamlit adding blank placeholder rows.
row_height_px = 34
header_height_px = 40
extra_padding_px = 6
table_height = int(header_height_px + extra_padding_px + max(1, len(display_df)) * row_height_px)
st.dataframe(
    styler,
    width='stretch',
    height=table_height,
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
