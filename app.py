import streamlit as st

st.set_page_config(page_title="Financial State Tracker", layout="wide")

pages = [
    st.Page("pages/00_Giới_thiệu_sản_phẩm.py", title="GIỚI THIỆU"),
    st.Page("pages/01_Tổng_quan.py", title="TỔNG QUAN"),
    st.Page("pages/02_Ngành.py", title="NGÀNH"),
    st.Page("pages/03_Doanh_nghiệp.py", title="DOANH NGHIỆP"),
    st.Page("pages/04_Phương_pháp.py", title="PHƯƠNG PHÁP"),
    st.Page("pages/05_Tải_dữ_liệu.py", title="DỮ LIỆU"),
]

nav = st.navigation(pages)
nav.run()
