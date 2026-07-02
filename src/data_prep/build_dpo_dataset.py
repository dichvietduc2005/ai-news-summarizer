import os
import json
import random
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

print("Đang tải ViT5-Finetuned để tạo dữ liệu Rejected...")
device = "cuda" if torch.cuda.is_available() else "cpu"
vit5_name = "models/vit5-finetuned"
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
vit5_model = AutoModelForSeq2SeqLM.from_pretrained(vit5_name).to(device)

def clean_prompt(text):
    """Làm sạch các dấu chấm lơ lửng, khoảng trắng thừa"""
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def is_complete_sentence(text):
    if not text or len(text.strip()) == 0: return False
    valid_endings = ('.', '!', '?', '"', '\u201d', "'")
    return text.strip().endswith(valid_endings)

# =============================================
# 5 CHIẾN THUẬT TẠO REJECTED CHẤT LƯỢNG CAO
# =============================================

def reject_model_decent(text):
    """Loại 1 (25%): Bài văn 5-7 điểm — Model sinh đàng hoàng nhưng kém hơn người.
    ĐÂY LÀ LOẠI GIÁ TRỊ NHẤT cho DPO."""
    input_text = "vietnews: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = vit5_model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def reject_lan_man(text):
    """Loại 2 (30%): Lười biếng bê nguyên câu gốc, không biết chắt lọc.
    Chọn NGẪU NHIÊN vài câu rải rác trong bài thay vì chỉ lấy 3 câu đầu."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 5]
    if len(sentences) >= 4:
        picked = random.sample(sentences, min(random.randint(2, 4), len(sentences)))
        return '. '.join(picked) + '.'
    elif len(sentences) >= 2:
        return '. '.join(sentences[:2]) + '.'
    return text[:300] + '.'

def reject_cat_cut(chosen):
    """Loại 3 (20%): Viết dở dang, câu bị lửng lơ không có dấu chấm."""
    words = chosen.split()
    if len(words) > 10:
        cut_point = random.randint(int(len(words) * 0.4), int(len(words) * 0.7))
        bad_text = " ".join(words[:cut_point])
        return re.sub(r'[.!?,;:]+$', '', bad_text)
    return " ".join(words[:5])

def reject_ao_giac(text, chosen):
    """Loại 4 (15%): Ảo giác — Chèn thông tin giả mạo TINH VI vào giữa câu.
    Thay vì chèn cứng ở cuối, ta thay thế ngẫu nhiên một thực thể trong bản tóm tắt."""
    base = reject_model_decent(text)

    # Tạo các mẩu thông tin giả mạo đa dạng
    fake_insertions = [
        "theo thống kê mới nhất với hơn 12 triệu trường hợp được ghi nhận",
        "gây thiệt hại ước tính lên đến 500 tỷ đồng",
        "theo phát biểu của Tiến sĩ Nguyễn Văn Minh tại Đại học Harvard",
        "dự kiến sẽ hoàn thành vào tháng 2 năm 2099",
        "ảnh hưởng trực tiếp đến hơn 45 quốc gia trên toàn cầu",
        "với sự tài trợ 890 triệu USD từ Quỹ Tiền tệ Quốc tế",
    ]

    sentences = [s.strip() for s in base.split('.') if len(s.strip()) > 3]
    if len(sentences) >= 2:
        insert_pos = random.randint(0, len(sentences) - 1)
        sentences.insert(insert_pos, random.choice(fake_insertions).capitalize())
        return '. '.join(sentences) + '.'

    return base + '. ' + random.choice(fake_insertions).capitalize() + '.'

def reject_lap_tu(text):
    """Loại 5 (10%): Lặp từ — Bài thi 0 điểm, chỉ giữ lại một ít."""
    input_text = "vietnews: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = vit5_model.generate(input_ids, max_length=150, num_beams=2, repetition_penalty=1.0)
    return tokenizer. (outputs[0], skip_special_tokens=True).strip()

def create_rejected_sample(prompt, chosen):
    """Phân bổ tỷ lệ lỗi theo giá trị học thuật"""
    roll = random.random()

    if roll < 0.30:      
        return reject_lan_man(prompt), "lan_man"
    elif roll < 0.55:    
        return reject_model_decent(prompt), "model_decent"
    elif roll < 0.75:    
        return reject_cat_cut(chosen), "cat_cut"
    elif roll < 0.90:    
        return reject_ao_giac(prompt, chosen), "ao_giac"
    else:                
        return reject_lap_tu(prompt), "lap_tu"

def main():
    input_csv = "data/processed/summary_data.csv"
    output_jsonl = "data/processed/dpo_dataset.jsonl"
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Không tìm thấy {input_csv}. Kiểm tra lại!")
        return
        
    dpo_data = []
    error_stats = {"lan_man": 0, "model_decent": 0, "cat_cut": 0, "ao_giac": 0, "lap_tu": 0}

    print(f"Bắt đầu sinh 5000 dữ liệu DPO CHẤT LƯỢNG CAO...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if len(dpo_data) >= 5000: break

        text = clean_prompt(str(row['text']).strip())
        chosen = str(row['summary']).strip()

        if len(text.split()) < 50 or not is_complete_sentence(text): continue
        if len(chosen.split()) < 15 or not is_complete_sentence(chosen): continue

        rejected, error_type = create_rejected_sample(text, chosen)
        if chosen != rejected and len(rejected.split()) > 3:
            dpo_data.append({"prompt": text, "chosen": chosen, "rejected": rejected})
            error_stats[error_type] += 1

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ TẠO DỮ LIỆU HOÀN TẤT! Tổng cộng: {len(dpo_data)} cặp")
    print("📊 THỐNG KÊ PHÂN BỔ LOẠI LỖI:")
    for k, v in error_stats.items():
        pct = v / len(dpo_data) * 100 if len(dpo_data) > 0 else 0
        print(f"   {k}: {v} cặp ({pct:.1f}%)")

if __name__ == "__main__":
    main()
