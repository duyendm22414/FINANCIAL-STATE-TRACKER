import streamlit as st
import pandas as pd

from src.io import load_data
from src.ui import sidebar_filters, kpi, inject_css
from src.logic import state_vi, build_warning_text
from src.schema import COL_YEAR, COL_INDUSTRY, COL_TICKER, COL_NAME

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Tổng quan ngành", layout="wide")
inject_css()

data = load_data()
labeled = data["Nguyên tắc phân loại"]

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero">
      <h1>Tổng quan ngành</h1>
      <p>Xem phân bố trạng thái ngành và thay đổi trạng thái tài chính của doanh nghiệp.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if labeled.empty:
    st.error("Thiếu file 05D_financial_state_rule_labeled.csv trong outputs/tables.")
    st.stop()

# =========================
# Filters (keep your existing sidebar logic)
# =========================
year, industry, _ = sidebar_filters(labeled)

df = labeled[(labeled[COL_YEAR] == year) & (labeled[COL_INDUSTRY] == industry)].copy()
if df.empty:
    st.info("Không có dữ liệu cho bộ lọc đã chọn.")
    st.stop()

# =========================
# KPI row
# =========================
state_col = "Financial_State_Rule"
pass_col = "Pass_Ratio"

risk = df[state_col].astype(str).isin(["High_Risk", "At_Risk"]).mean()
healthy = (df[state_col].astype(str) == "Healthy").mean()

risk_tone = "danger" if risk >= 0.50 else ("warning" if risk >= 0.30 else "neutral")
healthy_tone = "success" if healthy >= 0.50 else ("info" if healthy >= 0.30 else "neutral")

c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.0])

with c1:
    kpi("Số doanh nghiệp", f"{df[COL_TICKER].nunique():,}", f"{industry} • {year}", tone="info")
with c2:
    kpi("Tỷ trọng rủi ro", f"{risk:.1%}", "Rủi ro cao & Nguy cơ rủi ro", tone=risk_tone)
with c3:
    kpi("Tỷ trọng khỏe mạnh", f"{healthy:.1%}", "Khỏe mạnh ", tone=healthy_tone)
with c4:
    if pass_col in df.columns:
        pass_med = pd.to_numeric(df[pass_col], errors="coerce").median()
        kpi("Tỷ lệ đạt chuẩn", "-" if pd.isna(pass_med) else f"{pass_med:.2f}", "Trung bình ngành", tone="neutral")
    else:
        kpi("Tỷ lệ đạt chuẩn", "-", "Không có tỷ lệ đạt chuẩn", tone="neutral")

st.write("")

# =========================
# Distribution + Warning / Watchlist (dashboard layout)
# =========================
left, right = st.columns([1.05, 1.15])

# ---------- LEFT: Distribution ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Phân bố trạng thái tài chính")

    # Count by state
    vc = df[state_col].astype(str).value_counts(dropna=False)

    dist = vc.reset_index()
    dist.columns = ["Mã trạng thái", "Số lượng"]
    dist["Trạng thái"] = dist["Mã trạng thái"].map(state_vi)

    order = ["High_Risk", "At_Risk", "Stable", "Healthy"]
    dist["Thứ tự"] = dist["Mã trạng thái"].apply(lambda x: order.index(x) if x in order else 99)
    dist = dist.sort_values("Thứ tự")

    # Bar chart
    chart_df = dist[["Trạng thái", "Số lượng"]].set_index("Trạng thái")
    st.bar_chart(chart_df)

    # Table
    st.dataframe(
        dist[["Trạng thái", "Số lượng"]],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Số lượng tính theo số dòng quan sát trong ngành-năm (không phải unique ticker nếu dữ liệu có lặp).")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: Warning + Watchlist ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tín hiệu thay đổi trạng thái tài chính")
    st.write(build_warning_text(df))
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Danh sách doanh nghiệp cần theo dõi")

    watch = df[df[state_col].astype(str).isin(["High_Risk", "At_Risk"])].copy()

    if watch.empty:
        st.info("Không có doanh nghiệp thuộc nhóm cần theo dõi trong ngành-năm này.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        watch["Trạng thái"] = watch[state_col].astype(str).map(state_vi)

        # Format pass ratio
        if pass_col in watch.columns:
            watch["_pass_num"] = pd.to_numeric(watch[pass_col], errors="coerce")
            watch["Tỷ lệ đạt chuẩn"] = watch["_pass_num"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        else:
            watch["_pass_num"] = pd.NA
            watch["Tỷ lệ đạt chuẩn"] = ""

        order2 = {"High_Risk": 0, "At_Risk": 1}
        watch["_state_rank"] = watch[state_col].map(order2).fillna(99)

        # Sort: risk first, then pass ratio ascending (worse first)
        watch = watch.sort_values(["_state_rank", "_pass_num"], ascending=[True, True])

        show_cols = [COL_TICKER, COL_NAME, "Tỷ lệ đạt chuẩn", "Trạng thái"]
        show = watch[show_cols].drop_duplicates(subset=[COL_TICKER]).head(50).copy()

        st.dataframe(show, use_container_width=True, hide_index=True, height=420)

        st.caption("Ưu tiên High Risk trước, sau đó theo Pass Ratio thấp hơn trước (nếu có).")
        st.markdown("</div>", unsafe_allow_html=True)
