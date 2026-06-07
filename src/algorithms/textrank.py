# Giao việc cho Thành viên 2

def process_textrank(raw_text: str, language: str = 'vi', max_words: int = 300) -> str:
    """
    Hàm chạy thuật toán TextRank để trích xuất các câu quan trọng.
    - Input: Bài báo thô (str), ngôn ngữ ('vi' hoặc 'en'), số từ tối đa (int)
    - Output: Văn bản đã được rút gọn (str) đảm bảo dưới max_words
    """
    # TODO: TV2 thực hiện:
    # 1. Tách câu (sentence tokenize) dựa theo language
    # 2. Vector hóa câu (TF-IDF hoặc Word2Vec)
    # 3. Tính ma trận Cosine Similarity
    # 4. Chạy PageRank/TextRank để lấy Top câu
    # 5. Ghép lại thành String
    pass
