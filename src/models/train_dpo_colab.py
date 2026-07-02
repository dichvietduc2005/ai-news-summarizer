# src/models/train_dpo_colab.py
# === VÁ LỖI UNIGRAM DÀNH CHO BẢN CŨ ===
import torch, gc
torch.cuda.empty_cache()
gc.collect()

import tokenizers.models
import transformers.models.t5.tokenization_t5 as t5_tok
_OrigUnigram = tokenizers.models.Unigram
def _FixedUnigram(vocab, **kwargs):
    if isinstance(vocab, dict):
        vocab = list(vocab.items())
    return _OrigUnigram(vocab, **kwargs)
tokenizers.models.Unigram = _FixedUnigram
t5_tok.Unigram = _FixedUnigram

# === TRAIN DPO ===
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import DPOTrainer
import shutil

def train_dpo():
    print("🚀 BẮT ĐẦU HUẤN LUYỆN DPO (BẢN STABLE)...")
    model_name = "/content/vit5-finetuned"
    dataset_path = "/content/drive/MyDrive/DoAnAI/dpo_dataset.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map={"": 0})

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q", "v"],
        lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    )

    dataset = load_dataset('json', data_files={'train': dataset_path})
    train_dataset = dataset['train']

    # Dùng lại TrainingArguments của bản ổn định
    training_args = TrainingArguments(
        output_dir="/content/vit5-dpo-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        optim="paged_adamw_32bit",
        fp16=True,
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_length=1024,
        peft_config=lora_config,
        is_encoder_decoder=True, # Ở trl 0.8.6 cờ này cứu rỗi tất cả!
    )

    print("Đang Train DPO... (Khoảng 1-2 tiếng ☕)")
    dpo_trainer.train()

    print("Đang lưu Mô hình DPO...")
    dpo_trainer.model.save_pretrained("/content/vit5-dpo-adapter-final")
    tokenizer.save_pretrained("/content/vit5-dpo-adapter-final")

    print("Đang đóng gói và gửi về Google Drive...")
    shutil.make_archive("/content/vit5-dpo-adapter-final", 'zip', "/content/vit5-dpo-adapter-final")
    
    # Dùng shutil.copy thay cho lệnh !cp của Colab để file .py không bị báo lỗi cú pháp
    shutil.copy("/content/vit5-dpo-adapter-final.zip", "/content/drive/MyDrive/DoAnAI/")

    print("🎉 XONG! File 'vit5-dpo-adapter-final.zip' đã hạ cánh an toàn!")

if __name__ == "__main__":
    train_dpo()
