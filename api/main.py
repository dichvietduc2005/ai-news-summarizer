from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import các hàm xử lý từ các thành viên trong nhóm
from src.data_prep.crawler import clean_text, detect_language  # Từ Anh Quốc (TV1)
from src.algorithms.textrank import process_textrank  # Từ Viết Đức (TV2)
from src.algorithms.knowledge_graph import extract_entities  # Từ Viết Đức (TV2)
from src.models.summarizer import generate_summary  # Từ Trọng Nghĩa (TV3)
from src.models.translator import translate_text  # Từ Trung Kiên (TV4)

app = FastAPI(
    title="AI Summarization & Translation API",
    description="API Gateway kết nối các luồng Tiền xử lý, TextRank, NER, Tóm tắt và Dịch thuật.",
    version="1.0.0",
)


# Định nghĩa Schema đầu vào bằng Pydantic
class ArticleRequest(BaseModel):
    raw_text: str
    language: str = "auto"  # Mặc định tự động nhận diện ngôn ngữ nếu truyền "auto"
    max_words: int = 700  # Thiết lập ngưỡng từ an toàn (SAFE_WORD_LIMIT) theo TextRank


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

        # Nhận diện ngôn ngữ tự động nếu yêu cầu là 'auto', ngược lại dùng ngôn ngữ chỉ định
        if request.language == "auto":
            lang = detect_language(cleaned_text)
        else:
            lang = request.language if request.language in ["vi", "en"] else "vi"

        # ---- Bước 2: Nén văn bản bằng Semantic TextRank (Hàm của Viết Đức) ----
        # Giúp cắt giảm độ dài văn bản gốc, tránh lỗi tràn 512 tokens khi đưa vào mô hình LLM
        extracted_text = process_textrank(
            cleaned_text, language=lang, max_words=request.max_words
        )

        # ---- Bước 3: Trích xuất thực thể NER (Hàm của Viết Đức) ----
        # Phục vụ cho việc vẽ Đồ thị tri thức (Knowledge Graph) ở Frontend
        entities = extract_entities(cleaned_text)

        # ---- Bước 4: Tóm tắt văn bản bằng ViT5 / mT5-XLSum (Hàm của Trọng Nghĩa) ----
        summary = generate_summary(extracted_text, language=lang)

        # ---- Bước 5: Mở rộng Dịch thuật bằng MarianMT (Hàm của Trung Kiên) ----
        # Phân luồng xử lý theo Trường hợp 1 (vi -> en) hoặc Trường hợp 2 (en -> vi)
        if lang == "vi":
            direction = "vi-en"
        else:
            direction = "en-vi"

        translation = translate_text(summary, model_type=direction)

        # ---- Bước 6: Trả kết quả chuẩn JSON về Frontend ----
        return {
            "status": "success",
            "detected_language": lang,
            "direction": direction,
            "data": {
                "summary": summary,
                "translation": translation,
                "entities": entities,
            },
        }

    except Exception as e:
        # Log lỗi hệ thống và trả về mã lỗi HTTP 500
        print(f"[API Error]: Đã xảy ra lỗi trong pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Bật server uvicorn chạy ở port 8000 để Frontend kết nối
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
