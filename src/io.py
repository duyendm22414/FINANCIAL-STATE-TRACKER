# src/io.py
from __future__ import annotations
import os
import pandas as pd
import streamlit as st

BASE_DIR = os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "outputs", "tables")

FILES = {
    "Nguyên tắc phân loại": os.path.join(OUT_DIR, "05D_financial_state_rule_labeled.csv"),
    "Chuẩn trung bình ngành": os.path.join(OUT_DIR, "05B_industry_year_benchmarks.csv"),
    "Chỉ số đại diện (PCA)": os.path.join(OUT_DIR, "representative_indicators_by_industry_year.csv"),
    "Theo dõi trạng thái tài chính doanh nghiệp qua mỗi năm": os.path.join(OUT_DIR, "06D4_table5_adjacent_year_transitions.csv"),
    "Xu hướng trạng thái tài chính (2020-2024)": os.path.join(OUT_DIR, "06B_table6_corporate_financial_state_2020_2024.csv"),
}

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    data = {k: read_csv(v) for k, v in FILES.items()}
    # chuẩn hóa năm nếu có
    for k in ["Nguyên tắc phân loại", "Chuẩn trung bình ngành", "Chỉ số đại diện (PCA)"]:
        if not data[k].empty and "Năm" in data[k].columns:
            data[k]["Năm"] = pd.to_numeric(data[k]["Năm"], errors="coerce").astype("Int64")
    return data