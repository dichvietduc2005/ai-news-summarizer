# src/models/train_translator.py
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

def train_translator(model_name="Helsinki-NLP/opus-mt-vi-en", data_path="data/processed/translation_data.csv", output_dir="./models/marianmt-finetuned-vi-en"):
    print(f"Đang chuẩn bị fine-tune mô hình dịch thuật {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Dữ liệu cần có 2 cột: 'vi_text' và 'en_text'
    dataset = load_dataset('csv', data_files={'train': data_path})
    dataset = dataset['train'].train_test_split(test_size=0.1)

    def preprocess_function(examples):
        inputs = examples["vi_text"]
        targets = examples["en_text"]
        
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        labels = tokenizer(text_target=targets, max_length=256, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["vi_text", "en_text"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # SacreBLEU yêu cầu target phải ở dạng list of lists
        decoded_labels = [[label] for label in decoded_labels]
        
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Bắt đầu quá trình huấn luyện dịch thuật...")
    trainer.train()
    
    trainer.save_model(output_dir)
    print(f"Đã lưu mô hình tinh chỉnh tại: {output_dir}")

if __name__ == "__main__":
    train_translator()