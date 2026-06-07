# Giao việc cho Thành viên 4 (Nhóm trưởng)
import streamlit as st
import requests

# Cấu hình trang
st.set_page_config(page_title="AI News Summarizer", layout="wide")

st.title("Hệ thống Tóm tắt & Dịch thuật Báo chí AI")

# TODO: TV4 thiết kế giao diện UI:
# 1. Tạo Text area để nhập bài báo
# 2. Tạo nút "Xử lý"
# 3. Gọi requests.post() tới localhost:8000/process
# 4. Hiển thị 2 cột: Bản tóm tắt và Bản dịch
# 5. (Nâng cao) Vẽ Knowledge Graph hiển thị lên web
