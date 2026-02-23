# app.py
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================
# App identity
# =========================================================
APP_NAME = "Financial State Tracker"
APP_SUBTITLE_EN = "HOSE Non-Financial Firms (2020–2024)"
APP_SUBTITLE_VI = "Doanh nghiệp phi tài chính HOSE (2020–2024)"

APP_TAGLINE_EN = (
    "Monitor financial states using PCA-based representative indicators, industry-year benchmarking, "
    "and a transparent pass-ratio state rule."
)
APP_TAGLINE_VI = (
    "Theo dõi trạng thái tài chính dựa trên chọn chỉ số đại diện bằng PCA, benchmark ngành-năm "
    "và quy tắc Pass Ratio minh bạch."
)

st.set_page_config(page_title=f"{APP_NAME} — HOSE", layout="wide")

# =========================================================
# Paths (match your repo structure)
# =========================================================
BASE_DIR = os.getcwd()
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_TABLE_DIR = os.path.join(BASE_DIR, "outputs", "tables")

FILES = {
    "labeled": os.path.join(OUT_TABLE_DIR, "05D_financial_state_rule_labeled.csv"),
    "pass_matrix": os.path.join(OUT_TABLE_DIR, "05C_pass_matrix_full_dataset.csv"),
    "benchmarks": os.path.join(OUT_TABLE_DIR, "05B_industry_year_benchmarks.csv"),
    "rep_ind": os.path.join(OUT_TABLE_DIR, "representative_indicators_by_industry_year.csv"),
    "pca_struct": os.path.join(OUT_TABLE_DIR, "pca_structure_by_industry_year.csv"),
    "base": os.path.join(OUT_TABLE_DIR, "05A_base_with_representatives.csv"),
    "final_clean": os.path.join(DATA_PROCESSED_DIR, "financial_ratios_final_clean.csv"),
}

FINANCIAL_INDUSTRIES = ["Ngân hàng", "Tài chính", "Không xác định"]

# Internal codes (do NOT change; used for computations)
STATE_ORDER = ["High_Risk", "At_Risk", "Stable", "Healthy"]
GROUPS = ["Liquidity", "Leverage", "Efficiency", "Profitability"]

GROUP_DIRECTION = {
    "Liquidity": "higher_better",
    "Leverage": "lower_better",
    "Efficiency": "higher_better",
    "Profitability": "higher_better",
}

META_COLS = [
    "Mã",
    "Tên công ty",
    "Sàn",
    "Ngành ICB - cấp 1",
    "Ngành ICB - cấp 2",
    "Ngành ICB - cấp 3",
    "Ngành ICB - cấp 4",
    "Năm",
]

# =========================================================
# Ratio groups used by the thesis product pipeline
# =========================================================
RATIO_GROUPS = {
    "Liquidity": ["Current_Ratio", "Quick_Ratio", "Cash_Ratio"],
    "Leverage": ["Debt_Equity", "Net_Leverage"],
    "Efficiency": ["Asset_Turnover", "Fixed_Asset_Turnover"],
    "Profitability": ["ROA", "ROE", "ROS"],
}

# =========================================================
# Display dictionaries (professional names)
# =========================================================
DISPLAY = {
    "vi": {
        "group": {
            "Liquidity": "Thanh khoản",
            "Leverage": "Đòn bẩy",
            "Efficiency": "Hiệu quả hoạt động",
            "Profitability": "Khả năng sinh lời",
        },
        "state": {
            "High_Risk": "Rủi ro cao",
            "At_Risk": "Rủi ro",
            "Stable": "Ổn định",
            "Healthy": "Khỏe mạnh",
        },
        "pass": {1: "Đạt", 0: "Không đạt"},
        "na": "Không có dữ liệu",
    },
    "en": {
        "group": {
            "Liquidity": "Liquidity",
            "Leverage": "Leverage",
            "Efficiency": "Operating efficiency",
            "Profitability": "Profitability",
        },
        "state": {
            "High_Risk": "High risk",
            "At_Risk": "At risk",
            "Stable": "Stable",
            "Healthy": "Healthy",
        },
        "pass": {1: "Pass", 0: "Fail"},
        "na": "NA",
    },
}

def gname(lang: str, group_code: str) -> str:
    return DISPLAY.get(lang, DISPLAY["vi"])["group"].get(group_code, group_code)

def sname(lang: str, state_code: str) -> str:
    return DISPLAY.get(lang, DISPLAY["vi"])["state"].get(state_code, state_code)

def pass_label(lang: str, x):
    if pd.isna(x):
        return DISPLAY[lang]["na"]
    try:
        return DISPLAY[lang]["pass"][int(x)]
    except Exception:
        return DISPLAY[lang]["na"]

# =========================================================
# i18n
# =========================================================
def T(lang: str) -> Dict[str, str]:
    if lang == "en":
        return {
            "nav": "Navigation",
            "home": "Overview",
            "firm": "Firm Explorer",
            "industry": "Industry View",
            "transition": "State Transitions",
            "method": "Method",
            "data": "Data & Coverage",
            "downloads": "Downloads",
            "mode": "Run mode",
            "demo": "Demo (use precomputed outputs)",
            "prod": "Production (recompute from final_clean)",
            "run": "Run recomputation",
            "note_demo": "Demo mode reads files from outputs/tables for fast access.",
            "note_prod": "Production mode recomputes PCA, benchmarks, pass ratio, and states from final_clean.",
            "err_no_outputs": "Required output files were not found. Please run your notebooks first.",
            "industry1": "Industry (ICB level 1)",
            "ticker": "Company (Name + Ticker)",
            "year": "Year",
            "search": "Search (ticker or company name)",
            "state": "Financial state",
            "passratio": "Pass Ratio",
            "dim_results": "Dimension results (firm vs benchmark)",
            "trend": "Trend over time",
            "industry_context": "Industry-year context",
            "benchmarks": "Benchmarks and representative indicators",
            "state_dist": "State distribution",
            "top_rank": "Top firms (Pass Ratio)",
            "transition_matrix": "Transition matrix",
            "sankey": "Sankey flow",
            "method_title": "Method explained (report-aligned)",
            "data_title": "Data & coverage",
            "downloads_title": "Download outputs",
            "definitions": "Definitions",
            "settings": "Settings",
            "what_can_do": "What you can do in this website",
            "company_profile": "Company profile",
            "dimension": "Dimension",
            "representative": "Representative indicator",
            "firm_value": "Firm value",
            "benchmark": "Benchmark (industry-year)",
            "result": "Result",
        }

    return {
        "nav": "Điều hướng",
        "home": "Tổng quan",
        "firm": "Tra cứu doanh nghiệp",
        "industry": "Góc nhìn ngành",
        "transition": "Chuyển dịch trạng thái",
        "method": "Phương pháp",
        "data": "Dữ liệu & phạm vi",
        "downloads": "Tải dữ liệu",
        "mode": "Chế độ chạy",
        "demo": "Demo (dùng output có sẵn)",
        "prod": "Chạy thật (tái tính từ final_clean)",
        "run": "Chạy tái tính toán",
        "note_demo": "Chế độ Demo đọc sẵn các file trong outputs/tables để chạy nhanh.",
        "note_prod": "Chế độ Chạy thật tái tính PCA, benchmark, pass ratio và trạng thái từ final_clean.",
        "err_no_outputs": "Không tìm thấy file output cần thiết. Bạn hãy chạy notebook trước.",
        "industry1": "Ngành (ICB cấp 1)",
        "ticker": "Doanh nghiệp (Tên + Mã)",
        "year": "Năm",
        "search": "Tìm kiếm (mã hoặc tên công ty)",
        "state": "Trạng thái tài chính",
        "passratio": "Pass Ratio",
        "dim_results": "Kết quả theo nhóm (DN so với benchmark)",
        "trend": "Xu hướng theo thời gian",
        "industry_context": "Bối cảnh ngành-năm",
        "benchmarks": "Benchmark và chỉ số đại diện",
        "state_dist": "Phân phối trạng thái",
        "top_rank": "Top doanh nghiệp (Pass Ratio)",
        "transition_matrix": "Ma trận chuyển dịch",
        "sankey": "Sankey luồng chuyển dịch",
        "method_title": "Giải thích phương pháp (bám sát báo cáo)",
        "data_title": "Dữ liệu & phạm vi",
        "downloads_title": "Tải các file output",
        "definitions": "Định nghĩa",
        "settings": "Cài đặt",
        "what_can_do": "Bạn có thể làm gì trong website này",
        "company_profile": "Hồ sơ doanh nghiệp",
        "dimension": "Nhóm chỉ số",
        "representative": "Chỉ số đại diện",
        "firm_value": "Giá trị doanh nghiệp",
        "benchmark": "Benchmark (ngành-năm)",
        "result": "Kết quả",
    }

# =========================================================
# IO helpers
# =========================================================
def ensure_dirs():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUT_TABLE_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def normalize_year(df: pd.DataFrame) -> pd.DataFrame:
    if "Năm" in df.columns:
        df["Năm"] = pd.to_numeric(df["Năm"], errors="coerce").astype("Int64")
    return df

def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in `{name}`: {missing}")
        st.stop()

def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def normalize_benchmark_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make benchmark CSV compatible across versions.
    Your current benchmark file uses Benchmark_Mean (no Benchmark_Method).
    App expects Benchmark_Value and Benchmark_Method.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "Benchmark_Mean" in out.columns and "Benchmark_Value" not in out.columns:
        out = out.rename(columns={"Benchmark_Mean": "Benchmark_Value"})
    if "BenchmarkMean" in out.columns and "Benchmark_Value" not in out.columns:
        out = out.rename(columns={"BenchmarkMean": "Benchmark_Value"})
    if "Benchmark" in out.columns and "Benchmark_Value" not in out.columns:
        out = out.rename(columns={"Benchmark": "Benchmark_Value"})

    if "Benchmark_Method" not in out.columns:
        out["Benchmark_Method"] = "mean"

    if "Benchmark_Value" in out.columns:
        out["Benchmark_Value"] = pd.to_numeric(out["Benchmark_Value"], errors="coerce")

    return out

# =========================================================
# Core method functions (compact, report-aligned)
# =========================================================
def run_pca_block(
    df_block: pd.DataFrame,
    cols: List[str],
    explained_threshold: float,
    min_obs: int,
):
    valid_cols = [c for c in cols if c in df_block.columns]
    if len(valid_cols) < 2:
        return None

    X = df_block[valid_cols].dropna()
    if X.shape[0] < min_obs:
        return None

    Xs = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(Xs)

    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    n_comp = int(np.argmax(cum >= explained_threshold) + 1)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_cols,
        columns=[f"PC{i+1}" for i in range(len(valid_cols))],
    )
    return {
        "n_obs": int(X.shape[0]),
        "pc1_var": float(explained[0]) if len(explained) else np.nan,
        "n_comp": n_comp,
        "loadings": loadings,
    }

def pca_select_representatives(
    df_final: pd.DataFrame,
    explained_threshold: float = 0.80,
    min_obs: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reps, summaries = [], []
    if df_final.empty:
        return pd.DataFrame(), pd.DataFrame()

    for (industry, year), sub in df_final.groupby(["Ngành ICB - cấp 1", "Năm"]):
        if pd.isna(year):
            continue
        for gcode, cols in RATIO_GROUPS.items():
            res = run_pca_block(sub, cols, explained_threshold, min_obs)
            if res is None:
                continue
            rep = res["loadings"]["PC1"].abs().idxmax()
            reps.append(
                {
                    "Ngành ICB - cấp 1": industry,
                    "Năm": int(year),
                    "Nhóm chỉ số": gcode,
                    "Chỉ số đại diện (theo PCA)": rep,
                }
            )
            summaries.append(
                {
                    "Ngành ICB - cấp 1": industry,
                    "Năm": int(year),
                    "Nhóm chỉ số": gcode,
                    "Số quan sát": res["n_obs"],
                    "Tỷ lệ phương sai PC1": round(res["pc1_var"], 4) if pd.notna(res["pc1_var"]) else np.nan,
                    "Số PC đạt ngưỡng": res["n_comp"],
                }
            )

    rep_df = pd.DataFrame(reps)
    summ_df = pd.DataFrame(summaries)

    if not rep_df.empty:
        rep_df = rep_df.sort_values(["Ngành ICB - cấp 1", "Năm", "Nhóm chỉ số"])
    if not summ_df.empty:
        summ_df = summ_df.sort_values(["Ngành ICB - cấp 1", "Năm", "Nhóm chỉ số"])

    return rep_df, summ_df

def build_base_with_reps(df_final: pd.DataFrame, rep_long: pd.DataFrame) -> pd.DataFrame:
    if df_final.empty or rep_long.empty:
        return pd.DataFrame()

    rep_wide = (
        rep_long.pivot_table(
            index=["Ngành ICB - cấp 1", "Năm"],
            columns="Nhóm chỉ số",
            values="Chỉ số đại diện (theo PCA)",
            aggfunc="first",
        )
        .reset_index()
    )

    for g in GROUPS:
        if g not in rep_wide.columns:
            rep_wide[g] = np.nan

    return df_final.merge(rep_wide, on=["Ngành ICB - cấp 1", "Năm"], how="inner")

def compute_benchmarks(
    base: pd.DataFrame,
    method: str = "mean",
    min_group_n: int = 10,
    winsor_p: Optional[float] = None,
) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()

    rows = []
    for (industry, year), g in base.groupby(["Ngành ICB - cấp 1", "Năm"]):
        if pd.isna(year):
            continue
        if g.shape[0] < min_group_n:
            continue

        for dim in GROUPS:
            indicator = g[dim].iloc[0] if dim in g.columns else np.nan
            if pd.isna(indicator) or indicator not in base.columns:
                continue

            vals = pd.to_numeric(g[indicator], errors="coerce")
            if winsor_p is not None:
                vals = winsorize_series(vals, p=winsor_p)

            bench_val = float(vals.median(skipna=True)) if method == "median" else float(vals.mean(skipna=True))
            rows.append(
                {
                    "Ngành ICB - cấp 1": industry,
                    "Năm": int(year),
                    "Nhóm chỉ số": dim,
                    "Indicator_Name": indicator,
                    "Benchmark_Value": bench_val,
                    "Benchmark_Method": method,
                    "Winsorize_p": (winsor_p if winsor_p is not None else np.nan),
                    "n_obs": int(vals.notna().sum()),
                }
            )

    return pd.DataFrame(rows)

def build_pass_matrix(base: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    if base.empty or bench.empty:
        return pd.DataFrame()

    bench = normalize_benchmark_schema(bench)

    need_cols = ["Ngành ICB - cấp 1", "Năm", "Nhóm chỉ số", "Indicator_Name", "Benchmark_Value"]
    missing = [c for c in need_cols if c not in bench.columns]
    if missing:
        st.error(f"Benchmark file missing columns: {missing}")
        st.stop()

    bmap = {
        (r["Ngành ICB - cấp 1"], int(r["Năm"]), r["Nhóm chỉ số"]): (r["Indicator_Name"], r["Benchmark_Value"])
        for _, r in bench.dropna(
            subset=["Ngành ICB - cấp 1", "Năm", "Nhóm chỉ số", "Indicator_Name", "Benchmark_Value"]
        ).iterrows()
    }

    out = []
    for _, row in base.iterrows():
        if pd.isna(row.get("Năm", np.nan)):
            continue

        o = {c: row.get(c, np.nan) for c in META_COLS}
        n_app = 0
        n_pass = 0

        for dim in GROUPS:
            key = (row.get("Ngành ICB - cấp 1", np.nan), int(row.get("Năm")), dim)
            if key not in bmap:
                o[f"{dim}_RepIndicator"] = np.nan
                o[f"{dim}_BenchmarkMean"] = np.nan
                o[f"{dim}_Value"] = np.nan
                o[f"{dim}_Pass"] = np.nan
                continue

            ind, bench_val = bmap[key]
            val = pd.to_numeric(row.get(ind, np.nan), errors="coerce")

            o[f"{dim}_RepIndicator"] = ind
            o[f"{dim}_BenchmarkMean"] = bench_val
            o[f"{dim}_Value"] = val

            if pd.isna(val) or pd.isna(bench_val):
                o[f"{dim}_Pass"] = np.nan
                continue

            n_app += 1
            if GROUP_DIRECTION[dim] == "higher_better":
                passed = int(val >= bench_val)
            else:
                passed = int(val <= bench_val)

            o[f"{dim}_Pass"] = passed
            n_pass += passed

        o["Num_Applicable_Indicators"] = n_app
        o["Num_Pass_Indicators"] = n_pass
        o["Pass_Ratio"] = (n_pass / n_app) if n_app > 0 else np.nan
        out.append(o)

    return pd.DataFrame(out)

def classify_state(pass_ratio: float):
    if pd.isna(pass_ratio):
        return np.nan
    if pass_ratio < 0.25:
        return "High_Risk"
    if pass_ratio < 0.50:
        return "At_Risk"
    if pass_ratio < 0.75:
        return "Stable"
    return "Healthy"

def label_financial_state(pass_df: pd.DataFrame) -> pd.DataFrame:
    if pass_df.empty:
        return pd.DataFrame()
    df = pass_df.copy()
    df["Financial_State_Rule"] = df["Pass_Ratio"].apply(classify_state)
    return df

def transition_from_labeled(labeled: pd.DataFrame, y0: int, y1: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = (
        labeled[labeled["Năm"] == y0][["Mã", "Financial_State_Rule"]]
        .rename(columns={"Financial_State_Rule": "State_From"})
        .dropna()
    )
    b = (
        labeled[labeled["Năm"] == y1][["Mã", "Financial_State_Rule"]]
        .rename(columns={"Financial_State_Rule": "State_To"})
        .dropna()
    )
    t = a.merge(b, on="Mã", how="inner").dropna()

    mat = (
        t.groupby(["State_From", "State_To"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    )
    return t, mat

# =========================================================
# Charts
# =========================================================
def fig_state_dist(df: pd.DataFrame, lang: str):
    d = df.copy()
    d["Financial_State_Rule"] = pd.Categorical(d["Financial_State_Rule"], categories=STATE_ORDER, ordered=True)
    g = d["Financial_State_Rule"].value_counts().reindex(STATE_ORDER).fillna(0).reset_index()
    g.columns = ["State", "Count"]
    g["State"] = g["State"].astype(str).map(lambda x: sname(lang, x))
    fig = px.bar(g, x="State", y="Count")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def fig_firm_trend_heatmap(firm_all: pd.DataFrame, lang: str):
    d = firm_all.copy()
    d = d.dropna(subset=["Năm"]).sort_values("Năm")
    if d.empty:
        return go.Figure()

    d["Năm"] = d["Năm"].astype(int)
    pivot = d.pivot_table(index=["Mã"], columns="Năm", values="Pass_Ratio", aggfunc="first").sort_index(axis=1)

    state_map = d.set_index("Năm")["Financial_State_Rule"].to_dict()
    years = list(pivot.columns)
    text = [[sname(lang, state_map.get(y, "NA")) for y in years]]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(y) for y in years],
            y=[d["Mã"].iloc[0]],
            text=text,
            texttemplate="%{text}<br>%{z:.2f}",
            hovertemplate="Year=%{x}<br>Pass Ratio=%{z:.3f}<br>State=%{text}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=("Year" if lang == "en" else "Năm"),
        yaxis_title=("Ticker" if lang == "en" else "Mã"),
    )
    return fig

def fig_firm_vs_benchmark(row: pd.Series, lang: str):
    records = []
    for dim in GROUPS:
        records.append(
            {
                "Dimension": gname(lang, dim),
                "Firm": row.get(f"{dim}_Value", np.nan),
                "Benchmark": row.get(f"{dim}_BenchmarkMean", np.nan),
            }
        )
    d = pd.DataFrame(records)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["Benchmark"],
            y=d["Firm"],
            mode="markers+text",
            text=d["Dimension"],
            textposition="top center",
        )
    )
    fig.update_layout(
        xaxis_title=("Industry-year benchmark" if lang == "en" else "Benchmark ngành-năm"),
        yaxis_title=("Firm value" if lang == "en" else "Giá trị doanh nghiệp"),
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def fig_passratio_trend(firm_all: pd.DataFrame, lang: str):
    d = firm_all.dropna(subset=["Năm"]).copy()
    if d.empty:
        return go.Figure()
    d["Năm"] = d["Năm"].astype(int)
    d = d.sort_values("Năm")
    fig = px.line(d, x="Năm", y="Pass_Ratio", markers=True)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=("Year" if lang == "en" else "Năm"),
        yaxis_title=("Pass Ratio" if lang == "en" else "Pass Ratio"),
    )
    return fig

def fig_transition_heatmap(mat: pd.DataFrame, lang: str):
    # show state names on axes
    m = mat.copy()
    m.index = [sname(lang, x) for x in m.index]
    m.columns = [sname(lang, x) for x in m.columns]
    fig = px.imshow(m, aspect="auto")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def fig_sankey(trans_table: pd.DataFrame, lang: str):
    df = trans_table.copy()
    df = df[df["State_From"].isin(STATE_ORDER) & df["State_To"].isin(STATE_ORDER)].copy()
    if df.empty:
        return go.Figure()

    left_nodes = [f"{sname(lang, s)} (t)" for s in STATE_ORDER]
    right_nodes = [f"{sname(lang, s)} (t+1)" for s in STATE_ORDER]
    nodes = left_nodes + right_nodes
    idx = {n: i for i, n in enumerate(nodes)}

    links = df.groupby(["State_From", "State_To"]).size().reset_index(name="value")
    source = [idx[f"{sname(lang, s)} (t)"] for s in links["State_From"]]
    target = [idx[f"{sname(lang, t)} (t+1)"] for t in links["State_To"]]
    value = links["value"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes, pad=14, thickness=14),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def fig_industry_avg_passratio(labeled: pd.DataFrame, industry: str, lang: str):
    d = labeled[labeled["Ngành ICB - cấp 1"] == industry].dropna(subset=["Năm"]).copy()
    if d.empty:
        return go.Figure()
    d["Năm"] = d["Năm"].astype(int)
    g = d.groupby("Năm")["Pass_Ratio"].mean().reset_index(name="Avg_Pass_Ratio").sort_values("Năm")
    fig = px.line(g, x="Năm", y="Avg_Pass_Ratio", markers=True)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=("Year" if lang == "en" else "Năm"),
        yaxis_title=("Average Pass Ratio" if lang == "en" else "Pass Ratio trung bình"),
    )
    return fig

def fig_industry_state_share(labeled: pd.DataFrame, industry: str, lang: str):
    d = labeled[labeled["Ngành ICB - cấp 1"] == industry].dropna(subset=["Năm"]).copy()
    if d.empty:
        return go.Figure()
    d["Năm"] = d["Năm"].astype(int)

    g = d.groupby(["Năm", "Financial_State_Rule"]).size().reset_index(name="Count")
    total = g.groupby("Năm")["Count"].transform("sum")
    g["Share"] = g["Count"] / total

    g["State_Label"] = g["Financial_State_Rule"].map(lambda x: sname(lang, str(x)))
    # keep consistent order
    g["Financial_State_Rule"] = pd.Categorical(g["Financial_State_Rule"], categories=STATE_ORDER, ordered=True)
    g = g.sort_values(["Năm", "Financial_State_Rule"])

    fig = px.bar(g, x="Năm", y="Share", color="State_Label", barmode="stack")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=("Year" if lang == "en" else "Năm"),
        yaxis_title=("Share" if lang == "en" else "Tỷ trọng"),
        legend_title=("State" if lang == "en" else "Trạng thái"),
    )
    return fig

# =========================================================
# Sidebar controls
# =========================================================
def sidebar_controls():
    st.sidebar.subheader("Settings / Cài đặt")

    lang_choice = st.sidebar.selectbox("Language / Ngôn ngữ", ["Tiếng Việt", "English"], index=0)
    lang = "vi" if lang_choice == "Tiếng Việt" else "en"
    t = T(lang)

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        t["nav"],
        [t["home"], t["firm"], t["industry"], t["transition"], t["method"], t["data"], t["downloads"]],
        index=0,
    )

    st.sidebar.markdown("---")
    mode = st.sidebar.radio(t["mode"], [t["demo"], t["prod"]], index=0)
    if mode == t["demo"]:
        st.sidebar.caption(t["note_demo"])
    else:
        st.sidebar.caption(t["note_prod"])

    prod_cfg = {
        "run_btn": False,
        "explained_threshold": 0.80,
        "min_obs": 10,
        "bench_method": "mean",
        "min_group_n": 10,
        "winsor_p": None,
    }

    if mode == t["prod"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Method parameters")
        prod_cfg["explained_threshold"] = st.sidebar.slider("Explained variance (cum.)", 0.60, 0.95, 0.80, 0.05)
        prod_cfg["min_obs"] = st.sidebar.number_input("Min obs per (industry, year)", 5, 200, 10, 1)

        st.sidebar.subheader("Benchmark")
        prod_cfg["bench_method"] = st.sidebar.selectbox("Benchmark method", ["mean", "median"], index=0)
        prod_cfg["min_group_n"] = st.sidebar.number_input("Min firms per (industry, year)", 5, 500, 10, 1)

        use_w = st.sidebar.checkbox("Winsorize", value=False)
        if use_w:
            prod_cfg["winsor_p"] = st.sidebar.slider("Winsorize p", 0.005, 0.05, 0.01, 0.005)

        prod_cfg["run_btn"] = st.sidebar.button(t["run"], type="primary", use_container_width=True)

    return lang, t, page, mode, prod_cfg

# =========================================================
# Data loading / recompute
# =========================================================
def load_demo_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = normalize_year(read_csv(FILES["labeled"]))
    pass_m = normalize_year(read_csv(FILES["pass_matrix"]))
    bench_raw = normalize_year(read_csv(FILES["benchmarks"]))
    bench = normalize_benchmark_schema(bench_raw)
    rep_ind = normalize_year(read_csv(FILES["rep_ind"]))
    pca_struct = normalize_year(read_csv(FILES["pca_struct"]))
    return labeled, pass_m, bench, rep_ind, pca_struct

def recompute_from_final_clean(
    prod_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_clean = normalize_year(read_csv(FILES["final_clean"]))
    if final_clean.empty:
        st.error("Missing data/processed/financial_ratios_final_clean.csv. Please generate it via notebooks.")
        st.stop()

    if "Ngành ICB - cấp 1" in final_clean.columns:
        final_clean = final_clean.loc[~final_clean["Ngành ICB - cấp 1"].isin(FINANCIAL_INDUSTRIES)].copy()

    rep_ind, pca_struct = pca_select_representatives(
        final_clean,
        explained_threshold=float(prod_cfg["explained_threshold"]),
        min_obs=int(prod_cfg["min_obs"]),
    )
    base = build_base_with_reps(final_clean, rep_ind)

    bench = compute_benchmarks(
        base,
        method=prod_cfg["bench_method"],
        min_group_n=int(prod_cfg["min_group_n"]),
        winsor_p=prod_cfg["winsor_p"],
    )
    pass_m = build_pass_matrix(base, bench)
    labeled = label_financial_state(pass_m)

    rep_ind.to_csv(FILES["rep_ind"], index=False, encoding="utf-8-sig")
    pca_struct.to_csv(FILES["pca_struct"], index=False, encoding="utf-8-sig")
    base.to_csv(FILES["base"], index=False, encoding="utf-8-sig")
    bench.to_csv(FILES["benchmarks"], index=False, encoding="utf-8-sig")
    pass_m.to_csv(FILES["pass_matrix"], index=False, encoding="utf-8-sig")
    labeled.to_csv(FILES["labeled"], index=False, encoding="utf-8-sig")

    return labeled, pass_m, bench, rep_ind, pca_struct

# =========================================================
# UI helpers (professional firm display)
# =========================================================
def firm_display_label(row: pd.Series) -> str:
    name = str(row.get("Tên công ty", "")).strip()
    ticker = str(row.get("Mã", "")).strip()
    if name and ticker:
        return f"{name} ({ticker})"
    if ticker:
        return ticker
    return name

def build_firm_options(labeled: pd.DataFrame) -> pd.DataFrame:
    df = labeled[["Mã", "Tên công ty", "Ngành ICB - cấp 1"]].drop_duplicates().copy()
    df["Firm_Label"] = df.apply(firm_display_label, axis=1)
    df["Firm_Label"] = df["Firm_Label"].astype(str)
    df = df.sort_values(["Ngành ICB - cấp 1", "Firm_Label"])
    return df

# =========================================================
# Pages
# =========================================================
def page_home(labeled: pd.DataFrame, lang: str, t: Dict[str, str]):
    st.title(APP_NAME)
    st.caption(APP_SUBTITLE_EN if lang == "en" else APP_SUBTITLE_VI)
    st.caption(APP_TAGLINE_EN if lang == "en" else APP_TAGLINE_VI)

    if labeled.empty:
        st.error(t["err_no_outputs"])
        return

    total_firms = int(labeled["Mã"].nunique())
    total_rows = int(labeled.shape[0])
    healthy_rows = int((labeled["Financial_State_Rule"] == "Healthy").sum())
    highrisk_rows = int((labeled["Financial_State_Rule"] == "High_Risk").sum())
    avg_pass = pd.to_numeric(labeled["Pass_Ratio"], errors="coerce").mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Firms" if lang == "en" else "Số DN", total_firms)
    c2.metric("Firm-year rows" if lang == "en" else "Số dòng firm-year", total_rows)
    c3.metric("Healthy rows" if lang == "en" else "Khỏe mạnh (dòng)", healthy_rows)
    c4.metric("High risk rows" if lang == "en" else "Rủi ro cao (dòng)", highrisk_rows)
    c5.metric("Avg Pass Ratio" if lang == "en" else "Pass Ratio TB", f"{avg_pass:.3f}" if pd.notna(avg_pass) else DISPLAY[lang]["na"])

    st.markdown("---")

    years = sorted([int(x) for x in labeled["Năm"].dropna().unique().tolist()])
    if not years:
        st.info("No year found in the output." if lang == "en" else "Không tìm thấy năm trong output.")
        return

    latest_year = years[-1]
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader(("Latest year overview" if lang == "en" else "Tổng quan năm mới nhất") + f" ({latest_year})")
        view = labeled[labeled["Năm"] == latest_year].copy()
        if view.empty:
            st.info("No data for the latest year." if lang == "en" else "Không có dữ liệu cho năm mới nhất.")
            return
        st.plotly_chart(fig_state_dist(view, lang), use_container_width=True)

        top = view.sort_values("Pass_Ratio", ascending=False).head(15).copy()
        top["State_Label"] = top["Financial_State_Rule"].map(lambda x: sname(lang, str(x)))
        st.subheader(t["top_rank"])
        st.dataframe(
            top[["Mã", "Tên công ty", "Ngành ICB - cấp 1", "Pass_Ratio", "State_Label"]],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader(t["what_can_do"])
        if lang == "en":
            st.markdown(
                """
- Explore a firm’s financial state and the reasons behind it (pass/fail by dimensions).
- Compare a firm’s values against the industry-year benchmarks.
- Examine how states migrate over time using a transition matrix and Sankey flow.
- Review the method in plain language aligned with the thesis report.
- Download the exact outputs used in the analysis.
                """
            )
        else:
            st.markdown(
                """
- Tra cứu trạng thái tài chính của một doanh nghiệp và lý do (đạt/không đạt theo từng nhóm).
- So sánh giá trị doanh nghiệp với benchmark ngành-năm.
- Quan sát chuyển dịch trạng thái theo thời gian qua ma trận và Sankey.
- Xem phương pháp giải thích rõ ràng bám theo báo cáo KLTN.
- Tải các file output đã dùng trong phân tích.
                """
            )

def firm_selector(labeled: pd.DataFrame, t: Dict[str, str], lang: str):
    industries = sorted(labeled["Ngành ICB - cấp 1"].dropna().unique().tolist())
    firm_opts = build_firm_options(labeled)

    c1, c2, c3, c4 = st.columns([2.0, 2.4, 1.1, 1.5])
    with c1:
        industry = st.selectbox(t["industry1"], industries)

    with c2:
        firm_in_ind = firm_opts[firm_opts["Ngành ICB - cấp 1"] == industry].copy()
        labels = firm_in_ind["Firm_Label"].tolist()
        chosen_label = st.selectbox(t["ticker"], labels)
        ticker = firm_in_ind.loc[firm_in_ind["Firm_Label"] == chosen_label, "Mã"].iloc[0]

    with c3:
        years_firm = sorted([int(x) for x in labeled[labeled["Mã"] == ticker]["Năm"].dropna().unique().tolist()])
        year = st.selectbox(t["year"], years_firm, index=len(years_firm) - 1 if years_firm else 0)

    with c4:
        search = st.text_input(t["search"], value="").strip()

    if search:
        s = search.upper()
        m1 = labeled["Mã"].astype(str).str.upper().str.contains(s, na=False)
        m2 = labeled["Tên công ty"].astype(str).str.upper().str.contains(s, na=False)
        matches = labeled[m1 | m2][["Mã", "Tên công ty", "Ngành ICB - cấp 1"]].drop_duplicates()

        if not matches.empty:
            ticker = matches.iloc[0]["Mã"]
            industry = matches.iloc[0]["Ngành ICB - cấp 1"]
            years_firm = sorted([int(x) for x in labeled[labeled["Mã"] == ticker]["Năm"].dropna().unique().tolist()])
            if years_firm:
                year = years_firm[-1]

    return industry, ticker, year

def page_firm(labeled: pd.DataFrame, lang: str, t: Dict[str, str]):
    st.header(t["firm"])
    st.caption(
        "Goal: present a clear state classification and the dimension-level explanation."
        if lang == "en"
        else "Mục tiêu: trình bày trạng thái tài chính rõ ràng và giải thích theo từng nhóm chỉ số."
    )

    if labeled.empty:
        st.error(t["err_no_outputs"])
        return

    _, ticker, year = firm_selector(labeled, t, lang)
    firm_all = labeled[labeled["Mã"] == ticker].sort_values("Năm").copy()
    firm_row = labeled[(labeled["Mã"] == ticker) & (labeled["Năm"] == year)].copy()

    st.markdown("---")
    st.subheader(t["company_profile"])
    prof = firm_all[META_COLS].drop_duplicates().head(1)
    st.dataframe(prof, use_container_width=True, hide_index=True)

    if firm_row.empty:
        st.info("No data for this firm-year." if lang == "en" else "Không có dữ liệu cho doanh nghiệp-năm này.")
        return

    row = firm_row.iloc[0]
    state_code = row.get("Financial_State_Rule", np.nan)
    pr = row.get("Pass_Ratio", np.nan)

    a, b, c = st.columns([1.2, 1.0, 2.2])
    a.metric(t["state"], sname(lang, str(state_code)) if pd.notna(state_code) else DISPLAY[lang]["na"])
    b.metric(t["passratio"], f"{float(pr):.3f}" if pd.notna(pr) else DISPLAY[lang]["na"])
    c.info(
        "Pass rule: Liquidity/Efficiency/Profitability pass if firm ≥ benchmark; Leverage passes if firm ≤ benchmark."
        if lang == "en"
        else "Quy tắc đạt: Thanh khoản/Hiệu quả/Sinh lời đạt nếu DN ≥ benchmark; Đòn bẩy đạt nếu DN ≤ benchmark."
    )

    st.subheader(t["dim_results"])
    records = []
    for dim in GROUPS:
        records.append(
            {
                t["dimension"]: gname(lang, dim),
                t["representative"]: row.get(f"{dim}_RepIndicator", np.nan),
                t["firm_value"]: row.get(f"{dim}_Value", np.nan),
                t["benchmark"]: row.get(f"{dim}_BenchmarkMean", np.nan),
                t["result"]: pass_label(lang, row.get(f"{dim}_Pass", np.nan)),
            }
        )

    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    st.plotly_chart(fig_firm_vs_benchmark(row, lang), use_container_width=True)

    st.subheader(t["trend"])
    st.plotly_chart(fig_passratio_trend(firm_all, lang), use_container_width=True)

    st.subheader("Trend table (whole period)" if lang == "en" else "Bảng xu hướng cả giai đoạn")
    st.plotly_chart(fig_firm_trend_heatmap(firm_all, lang), use_container_width=True)

    trend_tbl = firm_all[["Năm", "Pass_Ratio", "Financial_State_Rule"]].copy()
    if not trend_tbl.empty:
        trend_tbl["Năm"] = trend_tbl["Năm"].astype(int)
        trend_tbl["State_Label"] = trend_tbl["Financial_State_Rule"].map(lambda x: sname(lang, str(x)))
        trend_tbl = trend_tbl.sort_values("Năm")
    st.dataframe(trend_tbl[["Năm", "Pass_Ratio", "State_Label"]], use_container_width=True, hide_index=True)

    st.subheader(t["definitions"])
    if lang == "en":
        st.markdown(
            """
- Pass Ratio = number of passed dimensions / number of applicable dimensions.
- State rule:
  - High risk: Pass Ratio < 0.25
  - At risk: 0.25 ≤ Pass Ratio < 0.50
  - Stable: 0.50 ≤ Pass Ratio < 0.75
  - Healthy: Pass Ratio ≥ 0.75
            """
        )
    else:
        st.markdown(
            """
- Pass Ratio = số nhóm đạt / số nhóm có dữ liệu hợp lệ.
- Quy tắc trạng thái:
  - Rủi ro cao: Pass Ratio < 0.25
  - Rủi ro: 0.25 ≤ Pass Ratio < 0.50
  - Ổn định: 0.50 ≤ Pass Ratio < 0.75
  - Khỏe mạnh: Pass Ratio ≥ 0.75
            """
        )

def page_industry(labeled: pd.DataFrame, bench: pd.DataFrame, rep_ind: pd.DataFrame, lang: str, t: Dict[str, str]):
    st.header(t["industry"])
    st.caption(
        t["industry_context"]
        if lang == "en"
        else "Mục tiêu: cung cấp bối cảnh ngành-năm để hiểu benchmark và phân phối trạng thái."
    )

    if labeled.empty:
        st.error(t["err_no_outputs"])
        return

    industries = sorted(labeled["Ngành ICB - cấp 1"].dropna().unique().tolist())
    years_all = sorted([int(x) for x in labeled["Năm"].dropna().unique().tolist()])

    c1, c2 = st.columns([2, 1.2])
    with c1:
        industry = st.selectbox(t["industry1"], industries)
    with c2:
        year = st.selectbox(t["year"], years_all, index=len(years_all) - 1 if years_all else 0)

    view = labeled[(labeled["Ngành ICB - cấp 1"] == industry) & (labeled["Năm"] == year)].copy()

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader(t["state_dist"])
        if view.empty:
            st.info("No data for selected industry-year." if lang == "en" else "Không có dữ liệu cho ngành-năm đã chọn.")
        else:
            st.plotly_chart(fig_state_dist(view, lang), use_container_width=True)

        st.subheader(t["top_rank"])
        top = view.sort_values("Pass_Ratio", ascending=False).head(25) if not view.empty else pd.DataFrame()
        if top.empty:
            st.info("No ranking available." if lang == "en" else "Không có bảng xếp hạng.")
        else:
            top = top.copy()
            top["State_Label"] = top["Financial_State_Rule"].map(lambda x: sname(lang, str(x)))
            st.dataframe(
                top[["Mã", "Tên công ty", "Pass_Ratio", "State_Label"]],
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.subheader(t["benchmarks"])
        if bench.empty or rep_ind.empty:
            st.info("Benchmark files are missing." if lang == "en" else "Thiếu file benchmark/chỉ số đại diện.")
        else:
            bench = normalize_benchmark_schema(bench)

            b = bench[(bench["Ngành ICB - cấp 1"] == industry) & (bench["Năm"] == year)].copy()
            r = rep_ind[(rep_ind["Ngành ICB - cấp 1"] == industry) & (rep_ind["Năm"] == year)].copy()

            if not r.empty:
                r = r.copy()
                r["Nhóm hiển thị"] = r["Nhóm chỉ số"].map(lambda x: gname(lang, str(x)))
                st.caption("Representative indicators (PCA)" if lang == "en" else "Chỉ số đại diện (PCA)")
                st.dataframe(
                    r[["Nhóm hiển thị", "Chỉ số đại diện (theo PCA)"]].sort_values("Nhóm hiển thị"),
                    use_container_width=True,
                    hide_index=True,
                )

            if not b.empty:
                b = b.copy()
                b["Nhóm hiển thị"] = b["Nhóm chỉ số"].map(lambda x: gname(lang, str(x)))
                st.caption("Benchmarks (industry-year)" if lang == "en" else "Benchmark (ngành-năm)")
                show_cols = ["Nhóm hiển thị", "Indicator_Name", "Benchmark_Value", "n_obs", "Benchmark_Method"]
                show_cols = [c for c in show_cols if c in b.columns]
                st.dataframe(b[show_cols].sort_values("Nhóm hiển thị"), use_container_width=True, hide_index=True)
            else:
                st.info("No benchmark for selected industry-year." if lang == "en" else "Không có benchmark cho ngành-năm này.")

    st.markdown("---")
    st.subheader("Industry trend (whole period)" if lang == "en" else "Xu hướng ngành (cả giai đoạn)")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.caption("Average Pass Ratio by year" if lang == "en" else "Pass Ratio trung bình theo năm")
        st.plotly_chart(fig_industry_avg_passratio(labeled, industry, lang), use_container_width=True)

    with c2:
        st.caption("State share by year" if lang == "en" else "Tỷ trọng trạng thái theo năm")
        st.plotly_chart(fig_industry_state_share(labeled, industry, lang), use_container_width=True)

    trend_ind_tbl = (
        labeled[labeled["Ngành ICB - cấp 1"] == industry]
        .copy()
        .dropna(subset=["Năm"])
        .assign(Năm=lambda x: x["Năm"].astype(int))
        .groupby("Năm")
        .agg(
            Firms=("Mã", "nunique"),
            Avg_Pass_Ratio=("Pass_Ratio", "mean"),
            Healthy_Share=("Financial_State_Rule", lambda s: (s == "Healthy").mean()),
            HighRisk_Share=("Financial_State_Rule", lambda s: (s == "High_Risk").mean()),
        )
        .reset_index()
        .sort_values("Năm")
    )
    st.dataframe(trend_ind_tbl, use_container_width=True, hide_index=True)

def page_transition(labeled: pd.DataFrame, lang: str, t: Dict[str, str]):
    st.header(t["transition"])
    st.caption("Monitor migration across years." if lang == "en" else "Theo dõi sự dịch chuyển trạng thái qua các năm.")

    if labeled.empty:
        st.error(t["err_no_outputs"])
        return

    years = sorted([int(x) for x in labeled["Năm"].dropna().unique().tolist()])
    if len(years) < 2:
        st.info("Need at least 2 years." if lang == "en" else "Cần tối thiểu 2 năm.")
        return

    c1, c2 = st.columns(2)
    with c1:
        y0 = st.selectbox("From year" if lang == "en" else "Từ năm", years[:-1], index=0)
    with c2:
        y1 = st.selectbox("To year" if lang == "en" else "Đến năm", years[1:], index=len(years[1:]) - 1)

    trans_table, mat = transition_from_labeled(labeled, y0, y1)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader(t["transition_matrix"])
        st.dataframe(mat, use_container_width=True)
        st.plotly_chart(fig_transition_heatmap(mat, lang), use_container_width=True)

    with right:
        st.subheader(t["sankey"])
        st.plotly_chart(fig_sankey(trans_table, lang), use_container_width=True)

def page_method(lang: str, t: Dict[str, str]):
    st.header(t["method"])
    st.subheader(t["method_title"])

    if lang == "en":
        st.markdown(
            """
### Conceptual flow (the key thesis contribution)
1) Group ratios into four dimensions: Liquidity, Leverage, Efficiency, Profitability  
2) Representative selection by PCA (industry-year): choose the ratio with the largest absolute loading on PC1  
3) Industry-year benchmark: compute benchmark for the representative indicator (mean/median; optional winsorization)  
4) Pass rule:
   - Liquidity/Efficiency/Profitability pass if firm ≥ benchmark
   - Leverage passes if firm ≤ benchmark  
5) Pass Ratio: passed dimensions / applicable dimensions  
6) State rule:
   - High risk < 0.25
   - At risk [0.25, 0.50)
   - Stable [0.50, 0.75)
   - Healthy ≥ 0.75
            """
        )
    else:
        st.markdown(
            """
### Luồng logic (trọng tâm đóng góp trong KLTN)
1) Chia tỷ số thành 4 nhóm: Thanh khoản, Đòn bẩy, Hiệu quả hoạt động, Khả năng sinh lời  
2) PCA chọn chỉ số đại diện theo (ngành, năm): chọn tỷ số có |loading| lớn nhất trên PC1  
3) Benchmark ngành-năm: tính benchmark cho chỉ số đại diện (mean/median; có thể winsorize)  
4) Quy tắc đạt:
   - Thanh khoản/Hiệu quả/Sinh lời đạt nếu DN ≥ benchmark
   - Đòn bẩy đạt nếu DN ≤ benchmark  
5) Pass Ratio = số nhóm đạt / số nhóm có dữ liệu  
6) Quy tắc trạng thái:
   - Rủi ro cao < 0.25
   - Rủi ro [0.25, 0.50)
   - Ổn định [0.50, 0.75)
   - Khỏe mạnh ≥ 0.75
            """
        )

def page_data(labeled: pd.DataFrame, lang: str, t: Dict[str, str]):
    st.header(t["data"])
    st.subheader(t["data_title"])

    if labeled.empty:
        st.error(t["err_no_outputs"])
        return

    years = sorted([int(x) for x in labeled["Năm"].dropna().unique().tolist()])
    industries = sorted(labeled["Ngành ICB - cấp 1"].dropna().unique().tolist())
    firms = int(labeled["Mã"].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Years" if lang == "en" else "Năm", f"{years[0]}–{years[-1]}" if years else DISPLAY[lang]["na"])
    c2.metric("Industries" if lang == "en" else "Số ngành", len(industries))
    c3.metric("Firms" if lang == "en" else "Số DN", firms)

    st.markdown("---")
    st.subheader("Coverage by year" if lang == "en" else "Độ phủ theo năm")
    cov = labeled.groupby("Năm")["Mã"].nunique().reset_index(name="Firms")
    st.dataframe(cov, use_container_width=True, hide_index=True)

    st.subheader("Coverage by industry" if lang == "en" else "Độ phủ theo ngành")
    cov_i = (
        labeled.groupby("Ngành ICB - cấp 1")["Mã"]
        .nunique()
        .reset_index(name="Firms")
        .sort_values("Firms", ascending=False)
    )
    st.dataframe(cov_i, use_container_width=True, hide_index=True)

def page_downloads(
    labeled: pd.DataFrame,
    pass_m: pd.DataFrame,
    bench: pd.DataFrame,
    rep_ind: pd.DataFrame,
    pca_struct: pd.DataFrame,
    lang: str,
    t: Dict[str, str],
):
    st.header(t["downloads"])
    st.subheader(t["downloads_title"])

    # Normalize benchmark schema (safe)
    bench = normalize_benchmark_schema(bench)

    # Toggle: download filename style
    use_clean_filename = st.toggle(
        "Use professional file names for download" if lang == "en" else "Tải với tên file chuyên nghiệp",
        value=True,
    )

    # Professional labels shown on UI (do NOT change the internal file paths in FILES)
    download_items = [
        {
            "key": "labeled",
            "df": labeled,
            "ui_title_en": "Firm-year financial states (final)",
            "ui_desc_en": "Final output for each firm-year: pass ratio, dimension pass/fail, and financial state label.",
            "ui_title_vi": "Trạng thái tài chính theo DN-năm (bản cuối)",
            "ui_desc_vi": "Output cuối theo DN-năm: pass ratio, đạt/không đạt theo nhóm và nhãn trạng thái.",
            "clean_name": "financial_state_firm_year_classification.csv",
            "raw_name": os.path.basename(FILES["labeled"]),
        },
        {
            "key": "pass_matrix",
            "df": pass_m,
            "ui_title_en": "Dimension pass/fail matrix",
            "ui_desc_en": "Dimension-level comparison vs industry-year benchmarks and resulting pass ratio.",
            "ui_title_vi": "Ma trận đạt/không đạt theo nhóm",
            "ui_desc_vi": "So sánh theo từng nhóm với benchmark ngành-năm và pass ratio kết quả.",
            "clean_name": "financial_state_dimension_pass_matrix.csv",
            "raw_name": os.path.basename(FILES["pass_matrix"]),
        },
        {
            "key": "benchmarks",
            "df": bench,
            "ui_title_en": "Industry-year benchmarks",
            "ui_desc_en": "Benchmark values for the selected representative indicator in each industry-year.",
            "ui_title_vi": "Benchmark theo ngành-năm",
            "ui_desc_vi": "Giá trị benchmark cho chỉ số đại diện trong từng ngành-năm.",
            "clean_name": "financial_state_industry_year_benchmarks.csv",
            "raw_name": os.path.basename(FILES["benchmarks"]),
        },
        {
            "key": "rep_ind",
            "df": rep_ind,
            "ui_title_en": "PCA representative indicators (industry-year)",
            "ui_desc_en": "Representative indicator selected by PCA for each dimension in each industry-year.",
            "ui_title_vi": "Chỉ số đại diện PCA (ngành-năm)",
            "ui_desc_vi": "Chỉ số đại diện được PCA chọn cho từng nhóm trong từng ngành-năm.",
            "clean_name": "financial_state_pca_representative_indicators.csv",
            "raw_name": os.path.basename(FILES["rep_ind"]),
        },
        {
            "key": "pca_struct",
            "df": pca_struct,
            "ui_title_en": "PCA structure summary",
            "ui_desc_en": "PCA diagnostics: sample size, PC1 variance share, and number of PCs reaching the threshold.",
            "ui_title_vi": "Tóm tắt cấu trúc PCA",
            "ui_desc_vi": "Chẩn đoán PCA: số quan sát, tỷ lệ phương sai PC1, số PC đạt ngưỡng.",
            "clean_name": "financial_state_pca_structure_summary.csv",
            "raw_name": os.path.basename(FILES["pca_struct"]),
        },
    ]

    any_file = False

    for item in download_items:
        df = item["df"]
        if df is None or df.empty:
            continue

        any_file = True
        title = item["ui_title_en"] if lang == "en" else item["ui_title_vi"]
        desc = item["ui_desc_en"] if lang == "en" else item["ui_desc_vi"]
        out_name = item["clean_name"] if use_clean_filename else item["raw_name"]

        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(desc)

            st.download_button(
                label=("Download CSV" if lang == "en" else "Tải CSV"),
                data=to_csv_bytes(df),
                file_name=out_name,
                mime="text/csv",
                use_container_width=True,
            )

            # Optional: show underlying original filename (small, professional)
            st.caption(
                ("Source file: " + item["raw_name"]) if lang == "en" else ("File gốc: " + item["raw_name"])
            )

    if not any_file:
        st.info("No output available to download." if lang == "en" else "Không có output để tải.")

# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()

    lang, t, page, mode, prod_cfg = sidebar_controls()

    if mode == t["demo"]:
        labeled, pass_m, bench, rep_ind, pca_struct = load_demo_outputs()
        if labeled.empty:
            st.error(t["err_no_outputs"])
            return
    else:
        if prod_cfg["run_btn"]:
            with st.spinner("Recomputing outputs..." if lang == "en" else "Đang tái tính toán outputs..."):
                labeled, pass_m, bench, rep_ind, pca_struct = recompute_from_final_clean(prod_cfg)
        else:
            labeled, pass_m, bench, rep_ind, pca_struct = load_demo_outputs()
            if labeled.empty:
                st.error(t["err_no_outputs"])
                return

    require_cols(
        labeled,
        ["Mã", "Tên công ty", "Ngành ICB - cấp 1", "Năm", "Pass_Ratio", "Financial_State_Rule"],
        "05D_financial_state_rule_labeled.csv",
    )

    if page == t["home"]:
        page_home(labeled, lang, t)
    elif page == t["firm"]:
        page_firm(labeled, lang, t)
    elif page == t["industry"]:
        page_industry(labeled, bench, rep_ind, lang, t)
    elif page == t["transition"]:
        page_transition(labeled, lang, t)
    elif page == t["method"]:
        page_method(lang, t)
    elif page == t["data"]:
        page_data(labeled, lang, t)
    elif page == t["downloads"]:
        page_downloads(labeled, pass_m, bench, rep_ind, pca_struct, lang, t)

if __name__ == "__main__":
    main()