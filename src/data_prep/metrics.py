# Mở file: src/data_prep/metrics.py
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Đảm bảo NLTK có đủ dữ liệu tách từ
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def calculate_rouge(generated_summary: str, reference_summary: str) -> dict:
    """
    Hàm tính điểm ROUGE cho tóm tắt.
    - Output: Dictionary chứa điểm ROUGE-1, ROUGE-2, ROUGE-L (quy đổi ra thang 100)
    """
    if not generated_summary or not reference_summary:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Khởi tạo scorer cho 3 loại chỉ số cơ bản
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    # Lấy điểm fmeasure (trung bình hài hòa của Precision và Recall) và nhân 100 cho dễ đọc
    return {
        "rouge1": round(scores['rouge1'].fmeasure * 100, 2),
        "rouge2": round(scores['rouge2'].fmeasure * 100, 2),
        "rougeL": round(scores['rougeL'].fmeasure * 100, 2)
    }

def calculate_bleu(translated_text: str, reference_text: str) -> float:
    """
    Hàm tính điểm BLEU cho dịch thuật.
    - Output: Điểm BLEU từ 0.0 đến 100.0
    """
    if not translated_text or not reference_text:
        return 0.0

    # Tách chuỗi thành danh sách các từ (tokens) theo chuẩn đầu vào của NLTK BLEU
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]
    translated_tokens = nltk.word_tokenize(translated_text.lower())
    
    # Sử dụng SmoothingFunction để tránh điểm 0 tuyệt đối khi dịch câu quá ngắn không khớp n-gram
    smoothie = SmoothingFunction().method1
    
    score = sentence_bleu(
        reference_tokens, 
        translated_tokens, 
        smoothing_function=smoothie
    )
    
    return round(score * 100, 2)
