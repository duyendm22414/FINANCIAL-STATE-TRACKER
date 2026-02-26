import streamlit as st

st.set_page_config(page_title="Financial State Tracker", layout="wide")

pages = [
    st.Page("pages/00_Giới_thiệu_sản_phẩm.py", title="Giới thiệu sản phẩm"),
    st.Page("pages/01_Tổng_quan.py", title="Tổng quan"),
    st.Page("pages/02_Ngành.py", title="Ngành"),
    st.Page("pages/03_Doanh_nghiệp.py", title="Doanh nghiệp"),
    st.Page("pages/04_Phương_pháp.py", title="Phương pháp"),
    st.Page("pages/05_Tải_dữ_liệu.py", title="Tải dữ liệu"),
]

nav = st.navigation(pages)
nav.run()