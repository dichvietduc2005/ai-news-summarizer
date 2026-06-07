# Giao việc cho Thành viên 3

def load_summarization_model(model_name: str = "VietAI/vit5-base"):
    """
    Load mô hình tóm tắt và tokenizer.
    """
    # TODO: TV3 sử dụng transformers.AutoModelForSeq2SeqLM
    pass

def generate_summary(extracted_text: str, language: str = 'vi') -> str:
    """
    Hàm sinh bản tóm tắt từ văn bản đã cắt gọt.
    - Input: Văn bản cắt gọt (str), ngôn ngữ ('vi' hoặc 'en')
    - Output: Bản tóm tắt tóm lược (str)
    """
    # TODO: TV3 thực hiện:
    # 1. Thêm prefix (VD: "vietnews: ")
    # 2. Đưa qua mô hình
    # 3. Decode kết quả trả về String
    pass
