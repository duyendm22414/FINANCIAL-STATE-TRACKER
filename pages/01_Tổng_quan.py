import streamlit as st
import pandas as pd
from src.io import load_data
from src.ui import sidebar_filters, kpi
from src.logic import state_vi, build_warning_text
from src.schema import COL_YEAR, COL_INDUSTRY, COL_TICKER, COL_NAME

data = load_data()
labeled = data["Nguyên tắc phân loại"]

st.title("Tổng quan theo ngành")
from src.ui import inject_css
inject_css()
st.caption("Xem phân bố trạng thái trong ngành theo từng năm và tín hiệu thay đổi trạng thái tài chính doanh nghiệp.")
st.markdown(
    """
<style>
/* Table style */
.table-wrap { border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden; }
.table { width: 100%; border-collapse: collapse; background: white; }
.table th { text-align: left; font-size: 13px; padding: 10px 12px; background: #F8FAFC; border-bottom: 1px solid #E5E7EB; color: #0F172A; }
.table td { font-size: 13px; padding: 10px 12px; border-bottom: 1px solid #F1F5F9; color: #0F172A; }
.table tr:last-child td { border-bottom: none; }

/* State badge */
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 12px; border: 1px solid transparent; }
.badge-high { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }
.badge-atr  { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }
.badge-stb  { background: #EFF6FF; color: #1D4ED8; border-color: #BFDBFE; }
.badge-hlt  { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }

/* Number */
.num { font-variant-numeric: tabular-nums; }
.small { color: #64748B; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)
if labeled.empty:
    st.error("Thiếu file 05D_financial_state_rule_labeled.csv trong outputs/tables.")
    st.stop()

year, industry, _ = sidebar_filters(labeled)

df = labeled[(labeled[COL_YEAR] == year) & (labeled[COL_INDUSTRY] == industry)].copy()
if df.empty:
    st.info("Không có dữ liệu cho năm đã chọn.")
    st.stop()

# KPI
risk = df["Financial_State_Rule"].astype(str).isin(["High_Risk", "At_Risk"]).mean()
healthy = (df["Financial_State_Rule"].astype(str) == "Healthy").mean()

risk_tone = "danger" if risk >= 0.50 else ("warning" if risk >= 0.30 else "neutral")
healthy_tone = "success" if healthy >= 0.50 else ("info" if healthy >= 0.30 else "neutral")

c1, c2, c3 = st.columns(3)
with c1: 
    kpi("Số doanh nghiệp", f"{df[COL_TICKER].nunique():,}", f"{industry} • {year}", tone="info")
with c2: 
    kpi("Tỷ trọng rủi ro", f"{risk:.1%}", "Rủi ro cao + Nguy cơ rủi ro", tone=risk_tone)
with c3: 
    kpi("Tỷ trọng khỏe mạnh", f"{healthy:.1%}", "Khỏe mạnh", tone=healthy_tone)

st.markdown("---")

# Phân bố trạng thái
st.subheader("Phân bố trạng thái tài chính")

# Tạo bảng đếm trạng thái
vc = df["Financial_State_Rule"].astype(str).value_counts(dropna=False)
dist = vc.reset_index()
dist.columns = ["Mã trạng thái", "Số lượng"]

# Map sang tiếng Việt
dist["Trạng thái"] = dist["Mã trạng thái"].map(state_vi)

# Sắp xếp theo thứ tự logic
order = ["High_Risk", "At_Risk", "Stable", "Healthy"]
dist["Thứ tự"] = dist["Mã trạng thái"].apply(lambda x: order.index(x) if x in order else 99)
dist = dist.sort_values("Thứ tự")

# Chỉ hiển thị 2 cột cần thiết
st.dataframe(
    dist[["Trạng thái", "Số lượng"]],
    use_container_width=True,
    hide_index=True
)

st.subheader("Tín hiệu thay đổi trạng thái tài chính")
st.write(build_warning_text(df))

st.markdown("---")
st.subheader("Danh sách doanh nghiệp cần theo dõi")

# Lọc nhóm rủi ro
watch = df[df["Financial_State_Rule"].astype(str).isin(["High_Risk", "At_Risk"])].copy()

if watch.empty:
    st.info("Không có doanh nghiệp thuộc nhóm cần theo dõi trong ngành–năm này.")
else:
    # Map trạng thái sang tiếng Việt
    watch["Trạng thái"] = watch["Financial_State_Rule"].astype(str).map(state_vi)

    # Tạo cột Tỷ lệ đạt chuẩn và format %
    watch["Tỷ lệ đạt chuẩn"] = pd.to_numeric(watch["Pass_Ratio"], errors="coerce")
    watch["Tỷ lệ đạt chuẩn"] = watch["Tỷ lệ đạt chuẩn"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else ""
    )

    # Sắp xếp theo mức rủi ro rồi đến tỷ lệ đạt chuẩn tăng dần
    order = {"High_Risk": 0, "At_Risk": 1}
    watch["Thứ tự"] = watch["Financial_State_Rule"].map(order)

    show = (
        watch[[COL_TICKER, COL_NAME, "Tỷ lệ đạt chuẩn", "Trạng thái", "Thứ tự"]]
        .sort_values(["Thứ tự", "Tỷ lệ đạt chuẩn"], ascending=[True, True])
        .drop(columns=["Thứ tự"])
        .head(50)
    )

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True
    )

