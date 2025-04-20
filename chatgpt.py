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

# === STEP 1: ĐỌC FILE ===
def load_data():
    with open("ly.txt", "r", encoding="utf-8") as f:
        ly_texts = [line.strip() for line in f.readlines() if line.strip()]
    with open("hoa.txt", "r", encoding="utf-8") as f:
        hoa_texts = [line.strip() for line in f.readlines() if line.strip()]
    
    df = pd.DataFrame({
        "text": ly_texts + hoa_texts,
        "label": [0] * len(ly_texts) + [1] * len(hoa_texts)  # 0 = Vật lý, 1 = Hóa học
    })
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    return df

# === STEP 2: TOKENIZE ===
def preprocess_data(df):
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    def preprocess(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset, tokenizer

# === STEP 3: TÍNH ĐỘ CHÍNH XÁC ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# === STEP 4: HUẤN LUYỆN MÔ HÌNH ===
def train_model(dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir="./logs",  # Lưu log
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Tính độ chính xác
    )

    trainer.train()
    return model, tokenizer, trainer


# === STEP 5: DỰ ĐOÁN ===
def predict(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()
    return ("Hóa học" if label == 1 else "Vật lý", round(confidence, 4))

# === MAIN FUNCTION ===
def main():
    print("🚀 Đang tải dữ liệu từ ly.txt và hoa.txt...")
    df = load_data()
    print(f"✅ Dữ liệu gồm {len(df)} dòng.")

    print("🔄 Tiền xử lý và tokenizing...")
    dataset, tokenizer = preprocess_data(df)

    print("🏋️‍♂️ Đang huấn luyện mô hình PhoBERT...")
    model, tokenizer, trainer = train_model(dataset, tokenizer)

    print("📊 Đánh giá mô hình:")
    metrics = trainer.evaluate()
    print(metrics)

    while True:
        text = input("\nNhập câu cần phân loại ('q' để thoát): ")
        if text.lower() == "q":
            break
        label, confidence = predict(text, model, tokenizer)
        print(f"➡️ Kết quả: {label} (độ tin cậy: {confidence*100:.2f}%)")

if __name__ == "__main__":
    main()
