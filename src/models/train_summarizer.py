# src/models/train_summarizer.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np

def train_summarizer(model_name="VietAI/vit5-base", data_path="data/processed/summary_data.csv", output_dir="./models/vit5-finetuned"):
    print(f"Đang chuẩn bị fine-tune mô hình {model_name}...")
    
    # 1. Tải tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 2. Tải dữ liệu (giả sử file CSV có 2 cột: 'text' và 'summary')
    # TV1 (Anh Quốc) cần chuẩn bị file này
    dataset = load_dataset('csv', data_files={'train': data_path})
    
    # Chia tập train/val
    dataset = dataset['train'].train_test_split(test_size=0.1)

    # 3. Hàm tiền xử lý dữ liệu (Tokenization)
    def preprocess_function(examples):
        inputs = ["vietnews: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        
        # Tokenize nhãn (bản tóm tắt)
        labels = tokenizer(text_target=examples["summary"], max_length=150, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text", "summary"])

    # 4. Data Collator và Metric đánh giá ROUGE
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}

    # 5. Cấu hình Training Arguments (Tối ưu VRAM cho Nghĩa)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,   # Batch size nhỏ để tránh tràn VRAM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # Tích lũy gradient (tương đương batch_size = 16)
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Sử dụng Mixed Precision (float16) để giảm VRAM
        push_to_hub=False,
    )

    # 6. Khởi tạo Trainer và Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu quá trình huấn luyện...")
    trainer.train()
    
    # Lưu mô hình sau khi huấn luyện xong
    trainer.save_model(output_dir)
    print(f"Đã lưu mô hình tinh chỉnh tại: {output_dir}")

if __name__ == "__main__":
    # Cần đảm bảo file data/processed/summary_data.csv đã tồn tại do TV1 làm
    train_summarizer()