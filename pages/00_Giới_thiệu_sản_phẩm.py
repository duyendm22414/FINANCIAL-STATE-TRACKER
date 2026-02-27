import streamlit as st
from src.ui import inject_css

st.set_page_config(page_title="Financial State Tracker", layout="wide")
inject_css()

st.title("Financial State Tracker")
st.caption("Công cụ theo dõi trạng thái tài chính doanh nghiệp dựa vào chuẩn trung bình ngành trong giai đoạn 2020-2024.")
st.markdown(
    """
**Website này tập trung vào:**
- Trạng thái tài chính doanh nghiệp theo từng ngành (ICB cấp 1)
- Tiêu chuẩn phân loại minh bạch dựa trên quá trình phân tích thành phần chỉ số chính đại diện cho khả năng thanh khoản, đòn bẩy tài chính, hiệu quả hoạt động và khả năng sinh lời, từ đó so sánh với chuẩn trung bình ngành và tỷ lệ đạt chuẩn
- Ý nghĩa kinh tế và tín hiệu thay đổi trạng thái tài chính để hỗ trợ theo dõi và ra quyết định

**Bạn có thể làm gì ở đây?**
- **Xem tổng quan ngành:** trạng thái tài chính doanh nghiệp (Rủi ro cao / Nguy cơ rủi ro / Ổn định / Khỏe mạnh) và danh sách doanh nghiệp cần theo dõi.
- **Tra cứu doanh nghiệp:** xem **Trạng thái, Lý do (đạt/không đạt) và Xu hướng** trong giai đoạn 2020-2024.
- **Hiểu tiêu chuẩn phân loại:** xem **chỉ số đại diện** cho từng nhóm (theo ngành) và **chuẩn trung bình ngành** (Benchmark) để biết doanh nghiệp được đánh giá dựa trên tiêu chuẩn nào.
- **Nhận tín hiệu thay đổi trạng thái tài chính:** xác định ngành hoặc doanh nghiệp có dấu hiệu không ổn định để ưu tiên giám sát và phân tích hoặc ngược lại.

**Nguyên tắc đánh giá:**
- Doanh nghiệp được so sánh với **chuẩn trung bình ngành**.
- Khả năng thanh khoản / Hiệu quả hoạt động / Khả năng sinh lời: **đạt** nếu chỉ số của doanh nghiệp cao hơn chuẩn trung bình ngành.  
- Đòn bẩy: **đạt** nếu chỉ số của doanh nghiệp thấp hơn chuẩn trung bình ngành.  
- **Tỷ lệ đạt chuẩn** = Số nhóm tiêu chí đạt / Số nhóm có dữ liệu cho phép phân loại **trạng thái tài chính**.

"""
)
