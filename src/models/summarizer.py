# Giao việc cho Thành viên 3
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc

_models = {}
_tokenizers = {}

def get_device():
    """Kiểm tra và trả về thiết bị khả dụng (GPU hoặc CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_summarization_model(language: str = 'vi'):
    """
    Load mô hình tóm tắt và tokenizer.
    """
    global _models, _tokenizers
    
    if language in _models:
        return _models[language], _tokenizers[language]
        
    device = get_device()
    
    if language == 'vi':
        model_name = "./models/vit5-dpo-merged"
    elif language == 'en':
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
    else:
        raise ValueError("Chỉ hỗ trợ ngôn ngữ 'vi' (Tiếng Việt) hoặc 'en' (Tiếng Anh)")
        
    print(f"Đang tải mô hình {model_name} lên {device}...")
    
    torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    
    _models[language] = model
    _tokenizers[language] = tokenizer
    
    return model, tokenizer

def generate_summary(extracted_text: str, language: str = 'vi') -> str:
    """
    Hàm sinh bản tóm tắt từ văn bản đã cắt gọt.
    - Input: Văn bản cắt gọt (str), ngôn ngữ ('vi' hoặc 'en')
    - Output: Bản tóm tắt tóm lược (str)
    """
    model, tokenizer = load_summarization_model(language)
    device = get_device()
    
    if language == 'vi':
        input_text = "vietnews: " + extracted_text
    else:
        input_text = extracted_text
    
    inputs = tokenizer(
        input_text,
        max_length=512,    
        truncation=True,   
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,        
            min_length=30,           # Không ép model viết dài nữa để tránh lan man
            length_penalty=0.8,      # Phạt nếu viết dài (giúp model súc tích)
            num_beams=4,           
            early_stopping=True,     # Dừng ngay khi đủ ý
            no_repeat_ngram_size=3,  
            repetition_penalty=1.5   # Phạt nặng lặp từ
        )
        
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
    return summary
