import html
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI News Summarizer", layout="wide")
st.title("📰 Hệ thống Tóm tắt & Dịch thuật AI")
BACKEND_URL = "http://127.0.0.1:8000/process"

# --- HÀM TẠO NÚT COPY ĐỘC LẬP (KHÔNG LÀM SẬP REACT, KHÔNG RELOAD TRANG) ---
def custom_copy_button(text_to_copy, button_label):
    escaped_text = html.escape(text_to_copy)
    html_code = f"""
    <style>
        body {{ margin: 0; padding: 0; background-color: #0e1117; }}
        .copy-btn {{
            border: 1px solid #4B4B4B;
            border-radius: 0.5rem;
            padding: 0.4rem 0.8rem;
            background-color: #262730;
            color: #FAFAFA;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            font-family: "Source Sans Pro", sans-serif;
            margin-top: 5px;
        }}
        .copy-btn:hover {{
            border-color: #ff4b4b;
            color: #ff4b4b;
        }}
    </style>
    <div id="text-data" style="display: none;">{escaped_text}</div>
    <button class="copy-btn" onclick="copyToClipboard()">
        {button_label}
    </button>
    <script>
        function copyToClipboard() {{
            const text = document.getElementById("text-data").innerText;
            navigator.clipboard.writeText(text).then(() => {{
                alert("✅ Đã copy thành công!");
            }}).catch(err => {{
                // Phương án dự phòng nếu API Clipboard bị chặn
                const el = document.createElement('textarea');
                el.value = text;
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                alert("✅ Đã copy thành công!");
            }});
        }}
    </script>
    """
    # Chiều cao 45px vừa khít với nút bấm, không chiếm diện tích
    components.html(html_code, height=45)
# -------------------------------------------------------------------------

col_input, col_output = st.columns([1, 1])

# ================= CỘT 1: NHẬP LIỆU =================
with col_input:
    st.write("### 📥 1. Nhập nội dung bài báo")
    raw_text_input = st.text_area(
        "Dán toàn bộ văn bản (Tiếng Anh hoặc Tiếng Việt) vào đây:",
        height=300,
        placeholder="Ví dụ: Thủ tướng Chính phủ vừa ban hành nghị định mới...",
    )
    
    # NÚT COPY VĂN BẢN GỐC AN TOÀN
    if raw_text_input.strip():
        custom_copy_button(raw_text_input, "📋 Copy Văn Bản Gốc")

    with st.expander("⚙️ Cài đặt hệ thống (Nâng cao)"):
        c1, c2 = st.columns([1, 1])
        with c1:
            language_option = st.selectbox("Ngôn ngữ đầu vào:", ["auto", "vi", "en"])
        with c2:
            max_words_option = st.slider(
                "Ngưỡng từ an toàn TextRank:", 100, 1000, 700, 50
            )
            
        # Nút tick Test Mode (Rất hữu ích cho máy yếu)
        is_test_mode = st.checkbox("🛠️ Chế độ Máy Yếu (Bỏ qua mô hình Tóm tắt & Dịch để tránh tràn RAM)", value=False)

    st.write("### ⚙️ 2. Tuỳ chọn xử lý")

    action_mode = st.radio(
        "Chọn tác vụ bạn muốn thực hiện:",
        options=["summary_only", "summary_and_translate", "translate_only"],
        format_func=lambda x: {
            "summary_only": "📝 Chỉ tóm tắt văn bản",
            "summary_and_translate": "🌐 Tóm tắt & Dịch bản tóm tắt",
            "translate_only": "🔤 Chỉ dịch toàn bộ văn bản gốc (Không tóm tắt)"
        }[x]
    )

    translation_mode = "summary"
    if action_mode == "translate_only":
        translation_mode = "full_text"

    process_btn = st.button("🚀 Bắt đầu Xử lý", type="primary")


# ================= CỘT 2: KẾT QUẢ =================
with col_output:
    if "saved_result" not in st.session_state:
        st.session_state.saved_result = None

    header_placeholder = st.empty()
    header_placeholder.write("### 📊 Kết quả Xử lý")

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
                        "is_test_mode": is_test_mode,
                    }

                    response = requests.post(BACKEND_URL, json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.saved_result = {
                            "used_action_mode": action_mode,
                            "detected_lang": result.get("detected_language", "vi"),
                            "summary": result.get("data", {}).get("summary", ""),
                            "translation": result.get("data", {}).get("translation", ""),
                            "entities": result.get("data", {}).get("entities", []),
                            "graph_url": result.get("data", {}).get("graph_url", None)
                        }
                    else:
                        st.error(f"❌ Backend lỗi: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối Backend. Lỗi: {str(e)}")

    if st.session_state.saved_result is not None:
        res = st.session_state.saved_result
        text_to_copy = ""
        
        display_mode = res.get("used_action_mode", action_mode) 

        header_placeholder.markdown(
            f"### 📊 Kết quả Xử lý &nbsp;&nbsp;&nbsp; <span style='font-size: 16px; color: #28a745;'>🟢 Nhận diện ngôn ngữ: <b>{res['detected_lang'].upper()}</b></span>", 
            unsafe_allow_html=True
        )
        
        with st.container(height=450, border=True):
            if display_mode == "summary_only":
                st.info(f"📝 **Bản Tóm Tắt ({'Tiếng Việt' if res['detected_lang'] == 'vi' else 'Tiếng Anh'}):**")
                st.write(res['summary'])
                text_to_copy = res['summary']
                
            elif display_mode == "translate_only":
                st.info(f"🔤 **Bản Dịch Toàn Bộ Văn Bản Gốc:**")
                st.write(res['translation'])
                text_to_copy = res['translation']
                
            else:
                st.markdown("📝 **Bản Tóm Tắt:**")
                st.write(res['summary'])
                st.divider() 
                st.markdown("🌐 **Bản Dịch Tóm Tắt:**")
                st.write(res['translation'])
                text_to_copy = f"BẢN TÓM TẮT:\n{res['summary']}\n\nBẢN DỊCH TÓM TẮT:\n{res['translation']}"

        # NÚT COPY KẾT QUẢ ĐẦU RA AN TOÀN
        if text_to_copy:
            custom_copy_button(text_to_copy, "📋 Copy Kết Quả")
            
        graph_url = res.get("graph_url")
        if graph_url:
            # Nút bấm HTML/CSS cực xịn theo phong cách Material Design (Google) không dùng icon
            google_btn_html = f"""
            <a href="{graph_url}" target="_blank" style="
                display: inline-block;
                margin-top: 15px;
                padding: 10px 24px;
                background-color: #1a73e8;
                color: #ffffff;
                font-family: 'Google Sans', Roboto, Arial, sans-serif;
                font-size: 14px;
                font-weight: 500;
                text-decoration: none;
                border-radius: 4px;
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
                transition: background-color 0.2s, box-shadow 0.2s;
            " onmouseover="this.style.backgroundColor='#1765cc'; this.style.boxShadow='0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15)';" 
              onmouseout="this.style.backgroundColor='#1a73e8'; this.style.boxShadow='0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15)';">
                Mở Đồ thị Tri thức
            </a>
            """
            st.markdown(google_btn_html, unsafe_allow_html=True)
        
        with st.expander("📌 Xem các Thực thể trích xuất (NER)"):
            st.json(res['entities'])