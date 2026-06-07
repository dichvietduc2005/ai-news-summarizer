# Giao việc cho Thành viên 1

def calculate_rouge(generated_summary: str, reference_summary: str) -> dict:
    """
    Hàm tính điểm ROUGE cho tóm tắt.
    - Output: Dictionary chứa điểm ROUGE-1, ROUGE-2, ROUGE-L
    """
    # TODO: TV1 sử dụng thư viện rouge_score
    pass

def calculate_bleu(translated_text: str, reference_text: str) -> float:
    """
    Hàm tính điểm BLEU cho dịch thuật.
    """
    # TODO: TV1 sử dụng nltk.translate.bleu_score
    pass
