# src/schema.py
from __future__ import annotations

# Cột bắt buộc từ file 05D_financial_state_rule_labeled.csv
COL_TICKER = "Mã"
COL_NAME = "Tên công ty"
COL_INDUSTRY = "Ngành ICB - cấp 1"
COL_YEAR = "Năm"

COL_PASS_RATIO = "Pass_Ratio"
COL_STATE = "Financial_State_Rule"

GROUPS = ["Liquidity", "Leverage", "Efficiency", "Profitability"]

GROUP_VI = {
    "Liquidity": "Thanh khoản",
    "Leverage": "Đòn bẩy",
    "Efficiency": "Hiệu quả hoạt động",
    "Profitability": "Khả năng sinh lời",
}

STATE_VI = {
    "High_Risk": "Rủi ro cao",
    "At_Risk": "Nguy cơ rủi ro",
    "Stable": "Ổn định",
    "Healthy": "Khỏe mạnh",
}

# Quy tắc hướng tốt (để diễn giải)
DIRECTION_VI = {
    "Liquidity": "Chỉ số thanh khoản cao hơn trung bình ngành cho thấy doanh nghiệp quản lý tài chính hiệu quả.",
    "Efficiency": "Chỉ số hiệu quả hoạt động cao hơn trung bình ngành cho thấy doanh nghiệp sử dụng tài sản hiệu quả, quản lý tài sản tốt hơn để tạo ra doanh thu.",
    "Profitability": "Chỉ số sinh lời cao hơn trung bình ngành cho thấy doanh nghiệp có lợi thế cạnh tranh, hiệu quả hoạt động tốt, đáng được ưu tiên đầu tư.",
    "Leverage": "Chỉ số đòn bẩy tài chính thấp hơn trung bình ngành thể hiện sự thận trọng, cơ cấu vốn an toàn, khả năng trả nợ tốt.",
}

# Tên cột trong pass matrix (05C/05D tùy bạn)
def rep_col(dim: str) -> str:
    return f"{dim}_RepIndicator"

def bench_col(dim: str) -> str:
    return f"{dim}_BenchmarkMean"

def value_col(dim: str) -> str:
    return f"{dim}_Value"

def pass_col(dim: str) -> str:
    return f"{dim}_Pass"