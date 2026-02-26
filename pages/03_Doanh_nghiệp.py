import streamlit as st
import pandas as pd
from src.io import load_data
from src.ui import sidebar_filters, kpi
from src.logic import state_vi, why_failed_dimensions
from src.schema import (
    COL_YEAR, COL_INDUSTRY, COL_TICKER, COL_NAME,
    GROUPS, GROUP_VI, rep_col, bench_col, value_col, pass_col
)

data = load_data()
labeled = data["Nguyên tắc phân loại"]
table6 = data["table6"]

st.title("Trạng thái tài chính doanh nghiệp")

st.caption("Trình bày kết quả phân loại theo năm và giải thích đạt/không đạt theo các nhóm chỉ số về thanh khoản, đòn bẩy tài chính, hiệu quả hoạt dộng và khả năng sinh lời.")

from src.ui import inject_css, kpi
inject_css()

if labeled.empty:
    st.error("Thiếu file 05D_financial_state_rule_labeled.csv.")
    st.stop()

year, industry, ticker = sidebar_filters(labeled)

firm_row = labeled[(labeled[COL_TICKER] == ticker) & (labeled[COL_YEAR] == year)]
firm_all = labeled[labeled[COL_TICKER] == ticker].sort_values(COL_YEAR)

if firm_row.empty:
    st.info("Không có dữ liệu cho doanh nghiệp-năm đã chọn.")
    st.stop()

row = firm_row.iloc[0]
name = str(row.get(COL_NAME, "")).strip()

st.markdown(f"**{name} ({ticker})**  •  {industry}  •  Năm {year}")

# KPI
# KPI
state_code = str(row.get("Financial_State_Rule", "")).strip()
failed = why_failed_dimensions(row)  # list tiếng Việt (Thanh khoản, Đòn bẩy, ...)

# Pass Ratio -> %
pass_ratio = pd.to_numeric(row.get("Pass_Ratio", None), errors="coerce")
pass_ratio_txt = f"{pass_ratio:.1%}" if pd.notna(pass_ratio) else "NA"

# Tone theo trạng thái
tone_map = {
    "High_Risk": "danger",
    "At_Risk": "warning",
    "Stable": "info",
    "Healthy": "success",
}
tone_state = tone_map.get(state_code, "neutral")

# Tone cho nhóm chưa đạt
fail_count = len(failed) if failed else 0
tone_fail = "danger" if fail_count >= 3 else ("warning" if fail_count >= 1 else "success")

c1, c2, c3 = st.columns(3)

with c1:
    kpi(
        "Trạng thái tài chính",
        state_vi(state_code),
        "Phân loại theo tiêu chuẩn ngành-năm",
        tone=tone_state,
    )

with c2:
    kpi(
        "Tỷ lệ đạt chuẩn",
        pass_ratio_txt,
        "Số nhóm đạt / số nhóm có dữ liệu",
        tone="neutral",
    )

with c3:
    if failed:
        kpi(
            "Nhóm chưa đạt",
            f"{fail_count}/4",
            "Cần theo dõi: " + ", ".join(failed),
            tone=tone_fail,
        )
    else:
        kpi(
            "Nhóm chưa đạt",
            "0/4",
            "Đạt ở tất cả nhóm có dữ liệu",
            tone="success",
        )
st.markdown("---")
st.subheader("So sánh trung bình ngành")
records = []
for dim in GROUPS:
    records.append({
        "Nhóm chỉ số": GROUP_VI[dim],
        "Chỉ số đại diện": row.get(rep_col(dim)),
        "Giá trị DN": row.get(value_col(dim)),
        "Chuẩn ngành-năm": row.get(bench_col(dim)),
        "Kết quả": "Đạt" if str(row.get(pass_col(dim))) == "1" else ("Không đạt" if str(row.get(pass_col(dim))) == "0" else "Không đủ dữ liệu"),
    })
st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Xu hướng theo thời gian (2020-2024)")
trend = firm_all[[COL_YEAR, "Pass_Ratio", "Financial_State_Rule"]].dropna(subset=[COL_YEAR]).copy()
trend[COL_YEAR] = trend[COL_YEAR].astype(int)
trend["Trạng thái"] = trend["Financial_State_Rule"].astype(str).map(state_vi)
st.dataframe(trend[[COL_YEAR, "Pass_Ratio", "Trạng thái"]], use_container_width=True, hide_index=True)

# Nếu bạn muốn hiện “Overall State + Trend” từ bảng 06B (tổng hợp 2020-2024)
if table6 is not None and not table6.empty and "Symbol" in table6.columns:
    t6 = table6.copy()
    t6 = t6[t6["Symbol"].astype(str) == str(ticker)]
    if not t6.empty:
        st.markdown("---")
        st.subheader("Kết quả tổng hợp")
        st.dataframe(t6, use_container_width=True, hide_index=True)