# Giao việc cho Thành viên 4 (Nhóm trưởng)
from fastapi import FastAPI
from pydantic import BaseModel

# TODO: TV4 Import các hàm từ src.data_prep, src.algorithms, src.models

app = FastAPI(title="AI Summarization & Translation API")

class ArticleRequest(BaseModel):
    raw_text: str
    language: str = "vi"
    max_words: int = 300

@app.post("/process")
def process_article(request: ArticleRequest):
    """
    API chính nối toàn bộ luồng pipeline.
    """
    # TODO: TV4 ráp các khối lại với nhau
    # 1. extracted = process_textrank(...)
    # 2. summary = generate_summary(extracted)
    # 3. translation = translate_text(summary)
    # 4. entities = extract_entities(request.raw_text)
    
    return {
        "status": "success",
        "language": request.language,
        "data": {
            "summary": "Tóm tắt mẫu...",
            "translation": "Translation sample...",
            "entities": []
        }
    }
