import streamlit as st
from src.ui import inject_css

st.set_page_config(page_title="Financial State Tracker", layout="wide")
inject_css()

st.title("Financial State Tracker")
st.caption("Công cụ theo dõi trạng thái tài chính doanh nghiệp dựa vào chuẩn trung bình ngành trong giai đoạn 2020-2024.")
st.markdown(
    """
**Website này tập trung vào:**
- Trạng thái tài chính theo từng ngành (ICB cấp 1) và từng năm
- Tiêu chuẩn phân loại minh bạch dựa trên so sánh với chuẩn trung bình ngành và tỷ lệ đạt chuẩn
- Ý nghĩa kinh tế và tín hiệu thay đổi trạng thái tài chính để hỗ trợ theo dõi và ra quyết định

**Bạn có thể làm gì ở đây?**
- **Xem tổng quan ngành:** trạng thái tài chính doanh nghiệp (Rủi ro cao / Nguy cơ rủi ro / Ổn định / Khỏe mạnh) và danh sách doanh nghiệp cần theo dõi.
- **Tra cứu doanh nghiệp:** xem **Trạng thái + Lý do (đạt/không đạt theo tiêu chí phân loại) + Xu hướng** trong giai đoạn 2020-2024.
- **Hiểu tiêu chuẩn phân loại:** xem **chỉ số đại diện** cho từng nhóm (theo ngành) và **chuẩn trung bình ngành** (Benchmark) để biết doanh nghiệp được đánh giá dựa trên tiêu chuẩn nào.
- **Nhận tín hiệu thay đổi trạng thái tài chính:** xác định ngành hoặc doanh nghiệp có dấu hiệu không ổn định để ưu tiên giám sát và phân tích sâu.

**Nguyên tắc đánh giá (*Nghiên cứu cá nhân*)**
- Doanh nghiệp được so sánh với **chuẩn trung bình ngành**.
- Thanh khoản / Hiệu quả hoạt động / Khả năng sinh lời: **đạt** nếu chỉ số của doanh nghiệp cao hơn chuẩn trung bình ngành.  
- Đòn bẩy: **đạt** nếu chỉ số của doanh nghiệp thấp hơn chuẩn trung bình ngành.  
- **Tỷ lệ đạt chuẩn** = số nhóm đạt / số nhóm có dữ liệu cho phép phân loại **trạng thái tài chính**.

"""
)