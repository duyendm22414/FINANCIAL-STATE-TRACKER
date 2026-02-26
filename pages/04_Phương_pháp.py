import streamlit as st

st.title("Phương pháp phân loại trạng thái tài chính")
st.caption("Giải thích ngắn gọn, minh bạch, bám sát logic của mô hình.")

st.markdown(
    """
## 1) Tư duy đánh giá theo ngành-năm
Doanh nghiệp được đánh giá theo mức tương đối trong cùng ngành và cùng năm, vì mỗi ngành có đặc thù tài chính khác nhau.
Do đó, chuẩn so sánh sử dụng giá trị trung bình của ngành trong năm.

## 2) Chọn chỉ số đại diện (PCA)
Mỗi nhóm chỉ số (Thanh khoản, Đòn bẩy, Hiệu quả hoạt động, Khả năng sinh lời) có nhiều tỷ số gốc.
Để đảm bảo tính gọn và dễ diễn giải, hệ thống chọn 1 chỉ số đại diện cho mỗi nhóm theo từng ngành.

## 3) Quy tắc đạt chuẩn (so với chuẩn ngành-năm)
- Thanh khoản / Hiệu quả hoạt động / Khả năng sinh lời: đạt nếu doanh nghiệp cao hơn mức chuẩn trung bình ngành  
- Đòn bẩy: đạt nếu doanh nghiệp thấp hơn mức chuẩn trung bình ngành  

## 4) Tỷ lệ đạt chuẩn và phân loại 4 trạng thái
Tỷ lệ đạt chuẩn = số nhóm đạt / số nhóm có dữ liệu.
Từ đó phân loại thành 4 trạng thái:
- Rủi ro cao
- Nguy cơ rủi ro
- Ổn định
- Khỏe mạnh

## 5) Ý nghĩa cảnh báo
Trạng thái và tỷ lệ đạt chuẩn giúp:
- Theo dõi trạng thái doanh nghiệp qua từng thời kỳ 
- Phát hiện sớm doanh nghiệp có dấu hiệu suy yếu so với mặt bằng ngành
- Ưu tiên danh sách “cần theo dõi” theo ngành qua các năm
- Hỗ trợ ra quyết định theo dõi, đánh giá rủi ro và giám sát sức khỏe tài chính theo thời gian
"""
)