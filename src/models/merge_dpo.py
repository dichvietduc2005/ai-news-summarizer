# src/models/merge_dpo.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def merge_weights():
    # CHÚ Ý QUAN TRỌNG: Phải trỏ về model SFT của bạn, KHÔNG được dùng VietAI/vit5-base gốc
    base_model_name = "./models/vit5-finetuned" # (hoặc đường dẫn tới thư mục vit5-finetuned trên máy tính của bạn)
    adapter_dir = "./models/vit5-dpo-adapter-final" 
    output_dir = "./models/vit5-dpo-merged"
    
    print(f"Đang tải Base Model: {base_model_name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Đang kết hợp với Adapter LoRA tại: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    print("Đang nén và hợp nhất trọng số (Merge and Unload)...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Đang lưu mô hình hoàn chỉnh tại: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Hoàn tất! Bây giờ bạn có thể dùng mô hình tại ./models/vit5-dpo-merged")

if __name__ == "__main__":
    merge_weights()
