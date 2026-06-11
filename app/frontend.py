import streamlit as st
import requests

st.set_page_config(page_title="AI News Summarizer", layout="wide")
st.title("📰 Hệ thống Tóm tắt & Dịch thuật Báo chí AI")
st.subheader("Đồ án tốt nghiệp - Thành viên: Trung Kiên")

BACKEND_URL = "http://127.0.0.1:8000/process"

# ĐÃ SỬA: Xóa gap="large" để trên điện thoại không bị một khoảng trống quá xa
col_input, col_output = st.columns([1, 1])

# ================= CỘT 1: NHẬP LIỆU =================
with col_input:
    st.write("### 📥 1. Nhập nội dung bài báo")
    raw_text_input = st.text_area(
        "Dán toàn bộ văn bản (Tiếng Anh hoặc Tiếng Việt) vào đây:",
        height=250,
        placeholder="Ví dụ: Thủ tướng Chính phủ vừa ban hành nghị định mới...",
    )

    with st.expander("⚙️ Cài đặt hệ thống (Nâng cao)"):
        c1, c2 = st.columns([1, 1])
        with c1:
            language_option = st.selectbox("Ngôn ngữ đầu vào:", ["auto", "vi", "en"])
        with c2:
            max_words_option = st.slider("Ngưỡng từ an toàn TextRank:", 100, 1000, 700, 50)

    st.write("### ⚙️ 2. Tuỳ chọn xử lý")
    need_translation = st.checkbox("🌐 Kèm theo bản Dịch thuật (Anh <-> Việt)", value=False)

    process_btn = st.button("🚀 Bắt đầu Xử lý", type="primary")


# ================= CỘT 2: KẾT QUẢ =================
with col_output:
    # ĐÃ SỬA: Gói khối kết quả vào st.container() để ép layout bám sát lại trên Mobile
    with st.container():
        st.write("### 📊 Kết quả Xử lý")

        if process_btn:
            if not raw_text_input.strip():
                st.error("⚠️ Vui lòng dán nội dung bài báo ở cột bên trái trước!")
            else:
                with st.spinner("⏳ Hệ thống đang phân tích..."):
                    try:
                        payload = {
                            "raw_text": raw_text_input,
                            "language": language_option,
                            "max_words": max_words_option,
                        }
                        response = requests.post(BACKEND_URL, json=payload)

                        if response.status_code == 200:
                            result = response.json()
                            detected_lang = result.get("detected_language", "vi")
                            data = result.get("data", {})

                            summary = data.get("summary", "")
                            translation = data.get("translation", "")
                            entities = data.get("entities", [])

                            st.success(f"✅ Đã nhận diện ngôn ngữ: **{detected_lang.upper()}**")

                            if not need_translation:
                                st.info(f"📝 **Bản Tóm Tắt ({'Tiếng Việt' if detected_lang == 'vi' else 'Tiếng Anh'})**")
                                st.write(summary)
                            else:
                                tab1, tab2 = st.tabs(["📝 Bản Tóm Tắt", "🔤 Bản Dịch Thuật"])
                                with tab1:
                                    st.write(summary)
                                with tab2:
                                    st.write(translation)

                            with st.expander("📌 Xem các Thực thể trích xuất (NER)"):
                                st.json(entities)
                        else:
                            st.error(f"❌ Backend lỗi: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Lỗi kết nối Backend. Hãy chắc chắn Uvicorn đang chạy. Lỗi: {str(e)}")
        else:
            # Thông báo mặc định khi người dùng chưa bấm nút
            st.info("👈 Hãy dán bài báo và bấm nút Xử lý ở cột bên trái để xem kết quả tại đây.")