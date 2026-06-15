import streamlit as st
from src.ui import inject_css

# Cấu hình trang
st.set_page_config(page_title="📊 Financial State Tracker", layout="wide")
inject_css()

# Tiêu đề chính
st.title("📊 Financial State Tracker")
st.caption("Công cụ theo dõi trạng thái tài chính doanh nghiệp dựa vào chuẩn trung bình ngành (2020-2024).")

# Giới thiệu
st.header("🌟 Tổng quan")
st.markdown(
    """
Website này tập trung vào:
- **Trạng thái tài chính doanh nghiệp theo từng ngành (ICB cấp 1)**
- **Tiêu chuẩn phân loại minh bạch** dựa trên phân tích các chỉ số chính:
  - Khả năng thanh khoản  
  - Đòn bẩy tài chính  
  - Hiệu quả hoạt động  
  - Khả năng sinh lời  
- So sánh với **chuẩn trung bình ngành** và tỷ lệ đạt chuẩn
- Ý nghĩa kinh tế và tín hiệu thay đổi trạng thái tài chính để hỗ trợ ra quyết định
"""
)

st.divider()

# Các chức năng chính
st.header("🛠️ Bạn có thể làm gì ở đây?")
st.markdown(
    """
- 🔎 **Xem tổng quan ngành:** trạng thái tài chính (Rủi ro cao / Nguy cơ rủi ro / Ổn định / Khỏe mạnh) và danh sách doanh nghiệp cần theo dõi.  
- 🏢 **Tra cứu doanh nghiệp:** xem **Trạng thái, Lý do (đạt/không đạt) và Xu hướng** trong giai đoạn 2020-2024.  
- 📊 **Hiểu tiêu chuẩn phân loại:** xem **chỉ số đại diện** cho từng nhóm ngành và **chuẩn trung bình ngành** (Benchmark).  
- 🚨 **Nhận tín hiệu thay đổi:** xác định ngành/doanh nghiệp có dấu hiệu bất ổn để ưu tiên giám sát hoặc ngược lại.  
"""
)

st.divider()

# Nguyên tắc đánh giá
st.header("📏 Nguyên tắc đánh giá")
st.markdown(
    """
- Doanh nghiệp được so sánh với **chuẩn trung bình ngành**.  
- **Khả năng thanh khoản / Hiệu quả hoạt động / Khả năng sinh lời:** *đạt* nếu chỉ số cao hơn chuẩn trung bình ngành.  
- **Đòn bẩy tài chính:** *đạt* nếu chỉ số thấp hơn chuẩn trung bình ngành.  
- **Tỷ lệ đạt chuẩn** = Số nhóm tiêu chí đạt / Số nhóm có dữ liệu cho phép phân loại trạng thái tài chính.  
"""
)
