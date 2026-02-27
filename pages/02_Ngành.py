import streamlit as st
import pandas as pd

from src.io import load_data
from src.ui import sidebar_filters, inject_css
from src.schema import COL_YEAR, COL_INDUSTRY, GROUPS, GROUP_VI, DIRECTION_VI

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Ngành", layout="wide")
inject_css()

data = load_data()
labeled = data["Nguyên tắc phân loại"]
bench = data["Chuẩn trung bình ngành"]
rep = data["Chỉ số đại diện (PCA)"]

# =========================
# Header (dashboard style)
# =========================
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

# Filters (giữ nguyên logic sidebar)
year, industry, _ = sidebar_filters(labeled)

# =========================
# Top summary chips
# =========================
meta1, meta2, meta3 = st.columns([1.2, 2.0, 1.2])
with meta1:
    st.markdown('<div class="card"><b>Năm</b><br/>' + f"{year}" + "</div>", unsafe_allow_html=True)
with meta2:
    st.markdown('<div class="card"><b>Ngành (ICB cấp 1)</b><br/>' + f"{industry}" + "</div>", unsafe_allow_html=True)
with meta3:
    rep_ok = (rep is not None) and (not rep.empty)
    bench_ok = (bench is not None) and (not bench.empty)
    status = "Đủ dữ liệu" if (rep_ok and bench_ok) else "Thiếu một phần dữ liệu"
    st.markdown('<div class="card"><b>Trạng thái dữ liệu</b><br/>' + status + "</div>", unsafe_allow_html=True)

st.write("")

# =========================
# Section 1 & 2 in dashboard layout
# =========================
left, right = st.columns([1.0, 1.2])

# ---------- LEFT: Representative indicators ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Chỉ số tài chính đại diện (PCA)")

    r = (
        rep[(rep["Năm"] == year) & (rep["Ngành ICB - cấp 1"] == industry)].copy()
        if rep is not None and not rep.empty
        else pd.DataFrame()
    )

    if r.empty:
        st.info("Chưa có dữ liệu chỉ số đại diện cho ngành–năm này.")
    else:
        r["Nhóm"] = r["Nhóm chỉ số"].map(lambda x: GROUP_VI.get(str(x), str(x)))
        show_r = r[["Nhóm", "Chỉ số đại diện (theo PCA)"]].copy()
        show_r = show_r.sort_values("Nhóm")

        st.dataframe(show_r, use_container_width=True, hide_index=True)
        st.caption("Chỉ số đại diện được chọn theo PCA để đại diện cho từng nhóm chỉ số trong ngành–năm.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: Benchmarks ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Chuẩn trung bình ngành (Benchmark)")

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

        show_b = b[show_cols].copy().sort_values(["Nhóm", "Indicator_Name"])
        st.dataframe(show_b, use_container_width=True, hide_index=True, height=430)

        st.caption("*Benchmark_Mean là giá trị trung bình ngành-năm, dùng làm chuẩn so sánh tương đối khi đánh giá đạt/không đạt.*")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# =========================
# Section 3: Economic meaning (accordion)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("3) Ý nghĩa kinh tế và tiêu chuẩn đạt chuẩn")

st.caption("*Gợi ý diễn giải: mỗi nhóm chỉ số có hướng đánh giá khác nhau (cao hơn tốt hơn hoặc thấp hơn tốt hơn).*")

for dim in GROUPS:
    title = GROUP_VI.get(dim, dim)
    content = DIRECTION_VI.get(dim, "")
    with st.expander(title, expanded=(dim == GROUPS[0])):
        st.write(content)

st.markdown("</div>", unsafe_allow_html=True)
