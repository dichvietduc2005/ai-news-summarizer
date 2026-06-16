# Giao việc cho Thành viên 4 (Trung Kiên)
import torch
from transformers import MarianMTModel, MarianTokenizer
import gc
from src.algorithms.knowledge_graph import split_vietnamese_sentences  # Import hàm của Đức

# Biến toàn cục để Cache mô hình (Chỉ load 1 lần để tránh sập RAM)
_models = {}
_tokenizers = {}

def get_device():
    """Kiểm tra và trả về thiết bị khả dụng (GPU hoặc CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_translation_model(direction: str = "vi-en"):
    """
    Load mô hình dịch MarianMT tương ứng và lưu vào Cache.
    - Hỗ trợ 'vi-en' (Việt -> Anh) và 'en-vi' (Anh -> Việt)
    """
    global _models, _tokenizers
    
    # Nếu mô hình đã load rồi thì lấy thẳng từ Cache ra dùng
    if direction in _models:
        return _models[direction], _tokenizers[direction]
        
    device = get_device()
    model_name = f"Helsinki-NLP/opus-mt-{direction}"
    
    print(f"[Translator] Đang tải mô hình {model_name} lên {device}...")
    
    # Tải Tokenizer và Model từ HuggingFace
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model = model.to(device)
    
    # Lưu vào Cache
    _models[direction] = model
    _tokenizers[direction] = tokenizer
    
    return model, tokenizer

def translate_text(text: str, model_type: str = "vi-en", is_chunked: bool = False) -> str:
    """
    Dịch bản tóm tắt hoặc toàn bộ văn bản sang ngôn ngữ đích.
    - Input: Văn bản (str), Hướng dịch ('vi-en' hoặc 'en-vi'), is_chunked (bool)
    - Output: Bản dịch (str)
    """
    if not text or not text.strip():
        return ""
        
    model, tokenizer = load_translation_model(model_type)
    device = get_device()
    
    # TRƯỜNG HỢP 1: Dịch toàn bộ văn bản gốc (sử dụng kỹ thuật Chunking)
    if is_chunked:
        sentences = split_vietnamese_sentences(text)
        translated_sentences = []
        
        print(f"[Translator] Đang dịch văn bản gốc ({len(sentences)} câu)...")
        for sent in sentences:
            if not sent.strip():
                continue
                
            inputs = tokenizer(
                sent, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                translated_ids = model.generate(**inputs, max_length=512)
                
            out = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated_sentences.append(out)
            
        # Giải phóng VRAM
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
        return " ".join(translated_sentences)

    # TRƯỜNG HỢP 2: Dịch nhanh văn bản đã tóm tắt (dưới 512 tokens)
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)
    
    # Sinh văn bản dịch
    with torch.no_grad():
        translated_ids = model.generate(
            **inputs, 
            max_length=512,
            num_beams=4,  # Sử dụng Beam Search để câu dịch mượt mà hơn
            early_stopping=True
        )
        
    # Giải mã IDs thành văn bản text
    translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    # Giải phóng VRAM nếu dùng GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        
    return translation


def fine_tune_marian(dataset_path: str, direction: str = "vi-en"):
    """
    Huấn luyện tinh chỉnh (Fine-tune) mô hình MarianMT trên tập dữ liệu chuyên ngành.
    """
    print(f"[Fine-tune] Khởi động quá trình huấn luyện mô hình {direction}...")
    print(f"[Fine-tune] Load dữ liệu từ: {dataset_path}")
    pass