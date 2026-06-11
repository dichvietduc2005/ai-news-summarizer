# Giao việc cho Thành viên 4 (Trung Kiên)
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import gc

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

def translate_text(text: str, model_type: str = "vi-en") -> str:
    """
    Dịch bản tóm tắt sang ngôn ngữ đích (Chạy Zero-shot hiện tại).
    - Input: Bản tóm tắt (str), Hướng dịch ('vi-en' hoặc 'en-vi')
    - Output: Bản dịch (str)
    """
    if not text or not text.strip():
        return ""
        
    model, tokenizer = load_translation_model(model_type)
    device = get_device()
    
    # Mã hóa văn bản đầu vào
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
    (Khung code chuẩn bị sẵn, chờ TV1 cung cấp Dataset)
    """
    print(f"[Fine-tune] Khởi động quá trình huấn luyện mô hình {direction}...")
    print(f"[Fine-tune] Load dữ liệu từ: {dataset_path}")
    
    # TODO: Khi có file CSV/JSON từ Quốc, Kiên sẽ ráp code vào đây:
    # 1. raw_datasets = load_dataset('csv', data_files=dataset_path)
    # 2. Tokenize dataset
    # 3. Setup Seq2SeqTrainingArguments (learning_rate, batch_size, epochs...)
    # 4. trainer = Seq2SeqTrainer(...)
    # 5. trainer.train()
    # 6. trainer.save_model(f"./saved_models/marian-{direction}-finetuned")
    
    print("[Fine-tune] (Đang chờ Dataset để hoàn thiện...)")
    pass