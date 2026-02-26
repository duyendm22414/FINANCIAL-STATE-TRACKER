import streamlit as st
from src.io import load_data

st.title("Tải dữ liệu kết quả")
st.caption("Tải các bảng kết quả để kiểm tra, đối chiếu hoặc tích hợp.")

data = load_data()

for k, df in data.items():
    if df is None or df.empty:
        continue
    st.subheader(k)
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button(
        label=f"Tải {k}.csv",
        data=df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name=f"{k}.csv",
        mime="text/csv",
        use_container_width=True
    )