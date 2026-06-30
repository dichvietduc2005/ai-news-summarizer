import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import các hàm xử lý từ các thành viên trong nhóm
from src.data_prep.crawler import clean_text, detect_language  # Từ Anh Quốc (TV1)
from src.algorithms.textrank import process_textrank  # Từ Viết Đức (TV2)
from src.algorithms.knowledge_graph import extract_entities, build_knowledge_graph  # Từ Viết Đức (TV2)
from src.models.summarizer import generate_summary  # Từ Trọng Nghĩa (TV3)
from src.models.translator import translate_text  # Từ Trung Kiên (TV4)

app = FastAPI(
    title="AI Summarization & Translation API",
    description="API Gateway kết nối các luồng Tiền xử lý, TextRank, NER, Tóm tắt và Dịch thuật.",
    version="1.0.0",
)

# Tạo thư mục static để host file Đồ thị Tri thức
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Định nghĩa Schema đầu vào bằng Pydantic
class ArticleRequest(BaseModel):
    raw_text: str
    language: str = "auto"  # Mặc định tự động nhận diện ngôn ngữ nếu truyền "auto"
    max_words: int = 700  # Thiết luật ngưỡng từ an toàn (SAFE_WORD_LIMIT) theo TextRank
    translation_mode: str = (
        "summary"  # Chấp nhận "summary" hoặc "full_text" từ Postman/UI gửi lên
    )
    is_test_mode: bool = False  # Chế độ máy yếu (bỏ qua mô hình nặng)


@app.post("/process")
def process_article(request: ArticleRequest):
    """
    API chính chịu trách nhiệm tiếp nhận bài báo thô, làm sạch, định tuyến
    luồng xử lý theo ngôn ngữ (Anh/Việt) và trả về kết quả tóm tắt, dịch thuật, thực thể.
    """
    if not request.raw_text.strip():
        raise HTTPException(
            status_code=400, detail="Nội dung văn bản không được để trống."
        )

    try:
        # ---- Bước 1: Tiền xử lý & Làm sạch dữ liệu (Hàm của Anh Quốc) ----
        cleaned_text = clean_text(request.raw_text)

        # Chốt chặn bảo vệ nếu text rỗng
        if not cleaned_text or not cleaned_text.strip():
            cleaned_text = request.raw_text

        # Nhận diện ngôn ngữ tự động
        if request.language == "auto":
            # Gọi trực tiếp hàm ĐÃ HOÀN THIỆN của Quốc
            lang = detect_language(cleaned_text)

            # Chốt chặn cuối cùng phòng hờ hệ thống không nhận diện được
            if not lang or lang not in ["vi", "en"]:
                lang = "vi"
        else:
            lang = request.language if request.language in ["vi", "en"] else "vi"

        # ---- Bước 2: Nén văn bản bằng Semantic TextRank (Hàm của Viết Đức) ----
        extracted_text = process_textrank(
            cleaned_text, language=lang, max_words=request.max_words
        )

        # CHỐT CHẶN BẢO VỆ 3: Đề phòng TextRank trả về rỗng
        if not extracted_text:
            extracted_text = cleaned_text

        # ---- Bước 3: Trích xuất thực thể NER (Hàm của Viết Đức) ----
        entities = extract_entities(cleaned_text)
        if entities is None:
            entities = []
            
        # Tự động vẽ và lưu Đồ thị tri thức (Knowledge Graph)
        graph_url = None
        if len(entities) > 0:
            try:
                # Lưu đè file graph.html vào thư mục static (FastAPI sẽ serve thư mục này)
                build_knowledge_graph(entities, output_path="static/graph.html")
                graph_url = "http://127.0.0.1:8000/static/graph.html"
            except Exception as e:
                print(f"[KG Error]: Lỗi khi vẽ đồ thị: {e}")

        # ---- Bước 4: Tóm tắt văn bản bằng ViT5 / mT5-XLSum (Hàm THẬT của Trọng Nghĩa) ----
        if request.is_test_mode:
            summary = "[TEST MODE] Đây là bản tóm tắt giả lập. Đã bỏ qua mô hình ViT5 (900MB) để tránh quá tải RAM cho máy yếu. Thực thể và đồ thị vẫn hoạt động dựa trên văn bản gốc."
        else:
            summary = generate_summary(extracted_text, language=lang)

        # CHỐT CHẶN BẢO VỆ 4: Đề phòng mô hình tóm tắt bị trả về rỗng, lấy tạm extracted_text để Kiên dịch
        if not summary or not summary.strip():
            summary = extracted_text

        # ---- Bước 5: Mở rộng Dịch thuật bằng MarianMT (Hàm THẬT của Trung Kiên) ----
        if lang == "vi":
            direction = "vi-en"
        else:
            direction = "en-vi"

        if request.is_test_mode:
            translation = "[TEST MODE] This is a mock translation. Bypassed MarianMT model to save RAM."
        else:
            # Kiểm tra kịch bản kiểm thử từ trường translation_mode gửi lên
            if request.translation_mode == "full_text":
                # Kịch bản 2: Dịch toàn bộ văn bản gốc (sử dụng Chunking của Kiên)
                translation = translate_text(
                    cleaned_text, model_type=direction, is_chunked=True
                )
            else:
                # Kịch bản 1: Chỉ dịch bản tóm tắt ngắn (Luồng mặc định)
                translation = translate_text(
                    summary, model_type=direction, is_chunked=False
                )

        # ---- Bước 6: Trả kết quả chuẩn JSON về Frontend/Postman ----
        return {
            "status": "success",
            "detected_language": lang,
            "direction": direction,
            "translation_mode_executed": request.translation_mode,
            "data": {
                "summary": summary,
                "translation": translation,
                "entities": entities,
                "graph_url": graph_url
            },
        }

    except Exception as e:
        # Log lỗi hệ thống chi tiết ra Terminal console để MLOps dễ Debug
        print(f"[API Error]: Đã xảy ra lỗi trong pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Bật server uvicorn chạy ở port 8000 để Frontend/Postman kết nối
    # Chỉ reload khi có thay đổi trong thư mục api và src để tránh quét venv gây tràn tài nguyên Windows
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["api", "src"])
