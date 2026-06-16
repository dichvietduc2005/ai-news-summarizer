import requests
import streamlit as st

st.set_page_config(page_title="AI News Summarizer", layout="wide")
st.title("📰 Hệ thống Tóm tắt & Dịch thuật AI")
BACKEND_URL = "http://127.0.0.1:8000/process"

# Xóa gap="large" để trên điện thoại không bị một khoảng trống quá xa
col_input, col_output = st.columns([1, 1])

# ================= CỘT 1: NHẬP LIỆU =================
with col_input:
    st.write("### 📥 1. Nhập nội dung bài báo")
    raw_text_input = st.text_area(
        "Dán toàn bộ văn bản (Tiếng Anh hoặc Tiếng Việt) vào đây:",
        height=250,
        placeholder="Ví dụ: Thủ tướng Chính phủ vừa ban hành nghị định mới...",
    )

    # ĐÃ SỬA: Sửa lại thụt lề (indentation) cho đúng cấp bậc của with col_input
    with st.expander("⚙️ Cài đặt hệ thống (Nâng cao)"):
        c1, c2 = st.columns([1, 1])
        with c1:
            language_option = st.selectbox("Ngôn ngữ đầu vào:", ["auto", "vi", "en"])
        with c2:
            max_words_option = st.slider(
                "Ngưỡng từ an toàn TextRank:", 100, 1000, 700, 50
            )

    st.write("### ⚙️ 2. Tuỳ chọn xử lý")

    need_translation = st.checkbox(
        "🌐 Kèm theo bản Dịch thuật (Anh <-> Việt)", value=False
    )

    # Kiên (TV4) thêm: Lựa chọn luồng dịch theo yêu cầu
    translation_mode = "summary"
    if need_translation:
        translation_mode = st.radio(
            "Chọn nội dung cần dịch:",
            options=["summary", "full_text"],
            format_func=lambda x: (
                "Dịch bản tóm tắt" if x == "summary" else "Dịch toàn bộ văn bản gốc"
            ),
            horizontal=True,
        )

    process_btn = st.button("🚀 Bắt đầu Xử lý", type="primary")


# ================= CỘT 2: KẾT QUẢ =================
with col_output:
    # Gói khối kết quả vào st.container() để ép layout bám sát lại trên Mobile
    with st.container():
        st.write("### 📊 Kết quả Xử lý")

        # ĐÃ SỬA: Căn thẳng lề lại cho đúng vị trí bên trong 'with st.container()'
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
                            "translation_mode": translation_mode,
                        }

                        response = requests.post(BACKEND_URL, json=payload)

                        if response.status_code == 200:
                            result = response.json()
                            detected_lang = result.get("detected_language", "vi")
                            data = result.get("data", {})

                            summary = data.get("summary", "")
                            translation = data.get("translation", "")
                            entities = data.get("entities", [])

                            st.success(
                                f"✅ Đã nhận diện ngôn ngữ: **{detected_lang.upper()}**"
                            )

                            if not need_translation:
                                st.info(
                                    f"📝 **Bản Tóm Tắt ({'Tiếng Việt' if detected_lang == 'vi' else 'Tiếng Anh'})**"
                                )
                                st.write(summary)
                            else:
                                tab1, tab2 = st.tabs(
                                    ["📝 Bản Tóm Tắt", "🔤 Bản Dịch Thuật"]
                                )
                                with tab1:
                                    st.write(summary)
                                with tab2:
                                    st.write(translation)

                            with st.expander("📌 Xem các Thực thể trích xuất (NER)"):
                                st.json(entities)
                        else:
                            st.error(f"❌ Backend lỗi: {response.status_code}")
                    except Exception as e:
                        st.error(
                            f"❌ Lỗi kết nối Backend. Hãy chắc chắn Uvicorn đang chạy. Lỗi: {str(e)}"
                        )
