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

        /* ===== Layout ===== */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        /* ===== Card ===== */
        .card {
            background: #1E293B;
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            transition: 0.2s ease;
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.45);
        }

        /* ===== KPI ===== */
        .kpi {
            display: grid;
            gap: 6px;
        }

        .kpi .title {
            font-size: 0.9rem;
            color: #94A3B8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .kpi .value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #F1F5F9;
        }

        .kpi .sub {
            font-size: 0.9rem;
            color: #94A3B8;
        }

        /* ===== KPI Accent Colors ===== */
        .tone-neutral { border-left: 5px solid #64748B; }
        .tone-info    { border-left: 5px solid #3B82F6; }
        .tone-success { border-left: 5px solid #22C55E; }
        .tone-warning { border-left: 5px solid #F59E0B; }
        .tone-danger  { border-left: 5px solid #EF4444; }

        /* ===== Hero Section ===== */
        .hero {
            background: linear-gradient(135deg, #1E293B, #0F172A);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 28px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            color: #F8FAFC;
        }

        .hero p {
            margin-top: 8px;
            font-size: 1rem;
            color: #CBD5E1;
        }

        /* ===== DataFrame ===== */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background-color: #0B1220;
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        /* ===== Buttons ===== */
        .stButton>button {
            border-radius: 12px;
            padding: 0.5rem 1.2rem;
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
        <div class="card kpi {tone_class}">
            <div class="title">{title}</div>
            <div class="value">{value}</div>
            <div class="sub">{subtitle}</div>
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
