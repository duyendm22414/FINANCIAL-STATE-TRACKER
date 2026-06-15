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
st.title("📊 So sánh nội ngành và tiêu chuẩn phân loại")
st.caption("Đánh giá trạng thái theo mức tương đối cùng ngành, dựa trên chỉ số đại diện (PCA) và benchmark trung bình ngành.")

if labeled.empty:
    st.error("❌ Thiếu file 05D_financial_state_rule_labeled.csv.")
    st.stop()

year, industry, _ = sidebar_filters(labeled)

# Top summary cards
col1, col2, col3 = st.columns(3)
col1.metric("Năm", year)
col2.metric("Ngành (ICB cấp 1)", industry)
col3.metric(
    "Trạng thái dữ liệu",
    "Đủ dữ liệu" if ((rep is not None and not rep.empty) and (bench is not None and not bench.empty)) else "Thiếu một phần dữ liệu"
)

st.divider()

# 1) Representative indicators
st.subheader("1️⃣ Chỉ số tài chính đại diện (PCA)")
r = (
    rep[(rep["Năm"] == year) & (rep["Ngành ICB - cấp 1"] == industry)].copy()
    if rep is not None and not rep.empty
    else pd.DataFrame()
)
if r.empty:
    st.info("ℹ️ Chưa có dữ liệu chỉ số đại diện cho yêu cầu này.")
else:
    r["Nhóm"] = r["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))
    show_r = r[["Nhóm", "Chỉ số đại diện (theo PCA)"]].sort_values("Nhóm")
    st.dataframe(show_r, use_container_width=True, hide_index=True)
    st.caption("📌 Chỉ số đại diện được chọn theo PCA để phản ánh khả năng thanh khoản, đòn bẩy tài chính, hiệu quả hoạt động và khả năng sinh lời.")

st.divider()

# 2) Benchmarks
st.subheader("2️⃣ Chuẩn trung bình ngành (Benchmark)")
b = (
    bench[(bench["Năm"] == year) & (bench["Ngành ICB - cấp 1"] == industry)].copy()
    if bench is not None and not bench.empty
    else pd.DataFrame()
)
if b.empty:
    st.info("ℹ️ Chưa có dữ liệu chuẩn trung bình ngành cho yêu cầu này.")
else:
    b["Nhóm"] = b["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))
    if "Benchmark_Mean" not in b.columns:
        st.error("❌ File benchmark không có cột Benchmark_Mean. Vui lòng kiểm tra 05B_industry_year_benchmarks.csv")
        st.stop()
    show_cols = ["Nhóm", "Indicator_Name", "Benchmark_Mean"]
    if "n_obs" in b.columns:
        show_cols.append("n_obs")
    show_b = b[show_cols].sort_values(["Nhóm", "Indicator_Name"])
    st.dataframe(show_b, use_container_width=True, hide_index=True, height=430)
    st.caption("📌 Benchmark_Mean là mức trung bình ngành, dùng làm chuẩn so sánh tương đối khi đánh giá đạt/không đạt.")

st.divider()

# 3) Economic meaning
st.subheader("3️⃣ Ý nghĩa kinh tế và tiêu chuẩn đạt chuẩn")
st.caption("💡 Gợi ý diễn giải: mỗi nhóm chỉ số có hướng đánh giá khác nhau (cao hơn tốt hơn hoặc thấp hơn tốt hơn).")

for dim in GROUPS:
    title = GROUP_VI.get(dim, dim)
    content = DIRECTION_VI.get(dim, "")
    with st.expander(f"📂 {title}", expanded=(dim == GROUPS[0])):
        st.write(content)
