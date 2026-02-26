# src/ui.py
from __future__ import annotations
import streamlit as st

# =========================
# Inject CSS chung
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        /* Base typography spacing */
        .block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        p { color: #334155; }

        /* KPI card (refined) */
        .card {
            border-radius: 16px;
            padding: 18px 18px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
            min-height: 110px;
        }
        .card-title {
            font-size: 12px;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
            margin-bottom: 6px;
            font-variant-numeric: tabular-nums;
        }
        .card-sub {
            font-size: 13px;
            color: #64748B;
            line-height: 1.35;
        }

        /* Tone borders (subtle) */
        .tone-danger { border-color: #FECACA; background: #FEF2F2; }
        .tone-warning { border-color: #FDE68A; background: #FFFBEB; }
        .tone-success { border-color: #A7F3D0; background: #ECFDF5; }
        .tone-info { border-color: #BFDBFE; background: #EFF6FF; }
        .tone-neutral { border-color: #E5E7EB; background: #FFFFFF; }

        /* Section divider */
        .soft-divider { margin: 22px 0 18px; border-top: 1px solid #E5E7EB; }

        /* Pills list */
        .pill-wrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
        .pill {
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #E5E7EB;
            background: #F8FAFC;
            color: #0F172A;
            font-size: 12px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# KPI card có phân màu
# =========================
def kpi(title: str, value: str, subtitle: str = "", tone: str = "neutral"):
    tone_class = f"tone-{tone}" if tone in ["danger","warning","success","info","neutral"] else "tone-neutral"
    st.markdown(
        f"""
        <div class="card {tone_class}">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Sidebar filter (giữ nếu bạn đang dùng)
# =========================
def sidebar_filters(df):
    from src.schema import COL_YEAR, COL_INDUSTRY, COL_TICKER

    years = sorted(df[COL_YEAR].dropna().unique())
    industries = sorted(df[COL_INDUSTRY].dropna().unique())
    tickers = sorted(df[COL_TICKER].dropna().unique())

    year = st.sidebar.selectbox("Năm", years)
    industry = st.sidebar.selectbox("Ngành ICB - cấp 1", industries)
    ticker = st.sidebar.selectbox("Mã doanh nghiệp", tickers)

    return year, industry, ticker