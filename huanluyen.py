# train_and_save_model.py
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

def load_data():
    with open("ly.txt", "r", encoding="utf-8") as f:
        ly_texts = [line.strip() for line in f if line.strip()]
    with open("hoa.txt", "r", encoding="utf-8") as f:
        hoa_texts = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame({
        "text": ly_texts + hoa_texts,
        "label": [0]*len(ly_texts) + [1]*len(hoa_texts)
    }).sample(frac=1).reset_index(drop=True)
    return df

def preprocess_data(df, tokenizer):
    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_and_save_model():
    df = load_data()
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    dataset = preprocess_data(df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./model_output",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("✅ Mô hình đã được lưu vào thư mục 'saved_model'.")

if __name__ == "__main__":
    train_and_save_model()
