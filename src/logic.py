# src/logic.py
from __future__ import annotations
import pandas as pd
from .schema import GROUPS, STATE_VI, GROUP_VI, pass_col

def state_vi(state_code: str) -> str:
    return STATE_VI.get(str(state_code), str(state_code))

def group_vi(dim: str) -> str:
    return GROUP_VI.get(dim, dim)

def build_warning_text(df_ind_year: pd.DataFrame) -> str:
    """
    Cảnh báo theo ngành–năm: tỷ trọng DN ở trạng thái rủi ro (High_Risk + At_Risk).
    """
    if df_ind_year.empty or "Financial_State_Rule" not in df_ind_year.columns:
        return "Chưa đủ dữ liệu để tạo cảnh báo."

    s = df_ind_year["Financial_State_Rule"].astype(str)
    risk_share = (s.isin(["High_Risk", "At_Risk"]).mean())

    if risk_share >= 0.50:
        return "Cảnh báo: Tỷ trọng doanh nghiệp rủi ro trong ngành khá cao. Nên ưu tiên theo dõi thanh khoản, đòn bẩy và khả năng sinh lời."
    if risk_share >= 0.30:
        return "Lưu ý: Một phần đáng kể doanh nghiệp đang ở nhóm rủi ro. Nhà đầu tư/đơn vị quản trị cần theo dõi sát các nhóm chỉ số trọng yếu."
    return "Tín hiệu tích cực: Tỷ trọng doanh nghiệp rủi ro ở mức thấp hơn. Tuy nhiên vẫn cần theo dõi các doanh nghiệp có xu hướng suy giảm."

def why_failed_dimensions(row: pd.Series) -> list[str]:
    """
    Trả về danh sách nhóm chưa đạt chuẩn trong năm được chọn.
    """
    failed = []
    for dim in GROUPS:
        p = row.get(pass_col(dim))
        if pd.isna(p):
            continue
        try:
            if int(p) == 0:
                failed.append(group_vi(dim))
        except Exception:
            continue
    return failed