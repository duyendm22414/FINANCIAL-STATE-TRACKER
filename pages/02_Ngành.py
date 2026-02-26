import streamlit as st
from src.io import load_data
from src.ui import sidebar_filters
from src.schema import COL_YEAR, COL_INDUSTRY, GROUPS, GROUP_VI, DIRECTION_VI

data = load_data()
labeled = data["Nguyên tắc phân loại"]
bench = data["Chuẩn trung bình ngành"]
rep = data["Chỉ số đại diện (PCA)"]

st.title("So sánh nội ngành và tiêu chuẩn phân loại")
st.caption("Giải thích: đánh giá và theo dõi trạng thái theo mức tương đối cùng ngành, dựa trên chỉ số tài chính đại diện và phân loại theo chuẩn trung bình ngành.")

if labeled.empty:
    st.error("Thiếu file 05D_financial_state_rule_labeled.csv.")
    st.stop()

year, industry, _ = sidebar_filters(labeled)

st.markdown("---")
st.subheader("1) Chỉ số tài chính")
r = rep[(rep["Năm"] == year) & (rep["Ngành ICB - cấp 1"] == industry)].copy() if not rep.empty else None
if r is None or r.empty:
    st.info("Chưa có dữ liệu chỉ số đại diện cho ngành-năm này.")
else:
    r["Nhóm"] = r["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))
    st.dataframe(r[["Nhóm", "Chỉ số đại diện (theo PCA)"]], use_container_width=True, hide_index=True)

st.subheader("2) Chuẩn trung bình ngành (benchmark)")
b = bench[(bench["Năm"] == year) & (bench["Ngành ICB - cấp 1"] == industry)].copy() if not bench.empty else None

if b is None or b.empty:
    st.info("Chưa có dữ liệu chuẩn trung bình ngành cho ngành-năm này.")
else:
    b["Nhóm"] = b["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))

    if "Benchmark_Mean" not in b.columns:
        st.error("File benchmark không có cột Benchmark_Mean. Vui lòng kiểm tra 05B_industry_year_benchmarks.csv")
        st.stop()

    show_cols = ["Nhóm", "Indicator_Name", "Benchmark_Mean"]
    if "n_obs" in b.columns:
        show_cols.append("n_obs")

    st.dataframe(b[show_cols].sort_values("Nhóm"), use_container_width=True, hide_index=True)
st.markdown("---")
st.subheader("3) Ý nghĩa kinh tế và tiêu chuẩn đạt chuẩn")
for dim in GROUPS:
    st.markdown(f"**{GROUP_VI[dim]}**")
    st.write(DIRECTION_VI[dim])