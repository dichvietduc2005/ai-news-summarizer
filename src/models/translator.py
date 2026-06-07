# Giao việc cho Thành viên 4

def load_translation_model(direction: str = "vi-en"):
    """
    Load mô hình dịch MarianMT tương ứng.
    """
    # TODO: TV4 load 'Helsinki-NLP/opus-mt-vi-en' hoặc 'en-vi'
    pass

def fine_tune_marian(dataset_path: str, direction: str = "vi-en"):
    """
    Huấn luyện tinh chỉnh (Fine-tune) mô hình MarianMT trên tập dữ liệu chuyên ngành.
    - Input: Đường dẫn tới file dữ liệu song ngữ (CSV/JSON)
    """
    # TODO: TV4 thực hiện:
    # 1. Load pre-trained MarianMT
    # 2. Setup Seq2SeqTrainingArguments
    # 3. Tiến hành huấn luyện (Training)
    # 4. Lưu mô hình (save_pretrained)
    pass

def translate_text(text: str, direction: str = "vi-en") -> str:
    """
    Dịch bản tóm tắt sang ngôn ngữ đích.
    - Input: Bản tóm tắt (str), Hướng dịch ('vi-en' hoặc 'en-vi')
    - Output: Bản dịch (str)
    """
    # TODO: TV4 gọi hàm dịch (sử dụng model đã fine-tune nếu có, nếu không dùng zero-shot)
    pass
