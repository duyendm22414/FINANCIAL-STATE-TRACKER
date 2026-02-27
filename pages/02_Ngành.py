import streamlit as st
import pandas as pd

from src.io import load_data
from src.ui import sidebar_filters, inject_css
from src.schema import GROUPS, GROUP_VI, DIRECTION_VI

st.set_page_config(page_title="Ngành", layout="wide")
inject_css()

data = load_data()
labeled = data["Nguyên tắc phân loại"]
bench = data["Chuẩn trung bình ngành"]
rep = data["Chỉ số đại diện (PCA)"]

# Header
st.markdown(
    """
    <div class="hero">
      <h1>So sánh nội ngành và tiêu chuẩn phân loại</h1>
      <p>Đánh giá trạng thái theo mức tương đối cùng ngành, dựa trên chỉ số đại diện (PCA) và benchmark trung bình ngành.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if labeled.empty:
    st.error("Thiếu file 05D_financial_state_rule_labeled.csv.")
    st.stop()

year, industry, _ = sidebar_filters(labeled)

# Top summary cards (HTML 1 lần, không bọc widget)
st.markdown(
    f"""
    <div style="display:grid; grid-template-columns: 1.1fr 1.8fr 1.1fr; gap:16px; margin-top:8px;">
      <div class="card">
        <div style="color:#94A3B8; font-weight:600; font-size:0.9rem;">Năm</div>
        <div style="font-size:1.2rem; font-weight:800; margin-top:4px;">{year}</div>
      </div>
      <div class="card">
        <div style="color:#94A3B8; font-weight:600; font-size:0.9rem;">Ngành (ICB cấp 1)</div>
        <div style="font-size:1.2rem; font-weight:800; margin-top:4px;">{industry}</div>
      </div>
      <div class="card">
        <div style="color:#94A3B8; font-weight:600; font-size:0.9rem;">Trạng thái dữ liệu</div>
        <div style="font-size:1.05rem; font-weight:800; margin-top:4px;">
          {"Đủ dữ liệu" if ((rep is not None and not rep.empty) and (bench is not None and not bench.empty)) else "Thiếu một phần dữ liệu"}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1.0, 1.2], gap="large")

# 1) Representative indicators
with left:
    with st.container():
        st.markdown(
            '<div class="section-title"><span class="section-number">1)</span> Chỉ số tài chính đại diện (PCA)</div>',
            unsafe_allow_html=True,
        )

        r = (
            rep[(rep["Năm"] == year) & (rep["Ngành ICB - cấp 1"] == industry)].copy()
            if rep is not None and not rep.empty
            else pd.DataFrame()
        )

        if r.empty:
            st.info("Chưa có dữ liệu chỉ số đại diện cho yêu cầu này.")
        else:
            r["Nhóm"] = r["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))
            show_r = r[["Nhóm", "Chỉ số đại diện (theo PCA)"]].sort_values("Nhóm")
            st.dataframe(show_r, use_container_width=True, hide_index=True)
            st.caption("Chỉ số đại diện được chọn theo PCA để đại diện cho từng nhóm chỉ số trong ngành–năm.")

# 2) Benchmarks
with right:
    with st.container():
        st.markdown(
            '<div class="section-title"><span class="section-number">2)</span> Chuẩn trung bình ngành (Benchmark)</div>',
            unsafe_allow_html=True,
        )

        b = (
            bench[(bench["Năm"] == year) & (bench["Ngành ICB - cấp 1"] == industry)].copy()
            if bench is not None and not bench.empty
            else pd.DataFrame()
        )

        if b.empty:
            st.info("Chưa có dữ liệu chuẩn trung bình ngành cho yêu cầu này.")
        else:
            b["Nhóm"] = b["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))

            if "Benchmark_Mean" not in b.columns:
                st.error("File benchmark không có cột Benchmark_Mean. Vui lòng kiểm tra 05B_industry_year_benchmarks.csv")
                st.stop()

            show_cols = ["Nhóm", "Indicator_Name", "Benchmark_Mean"]
            if "n_obs" in b.columns:
                show_cols.append("n_obs")

            show_b = b[show_cols].sort_values(["Nhóm", "Indicator_Name"])
            st.dataframe(show_b, use_container_width=True, hide_index=True, height=430)
            st.caption("Benchmark_Mean là trung bình ngành–năm, dùng làm chuẩn so sánh tương đối khi đánh giá đạt/không đạt.")

st.write("")

# 3) Economic meaning
with st.container():
    st.markdown(
        '<div class="section-title"><span class="section-number">3)</span> Ý nghĩa kinh tế và tiêu chuẩn đạt chuẩn</div>',
        unsafe_allow_html=True,
    )
    st.caption("Gợi ý diễn giải: mỗi nhóm chỉ số có hướng đánh giá khác nhau (cao hơn tốt hơn hoặc thấp hơn tốt hơn).")

    for dim in GROUPS:
        title = GROUP_VI.get(dim, dim)
        content = DIRECTION_VI.get(dim, "")
        with st.expander(title, expanded=(dim == GROUPS[0])):
            st.write(content)
