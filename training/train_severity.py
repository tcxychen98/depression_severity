import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# 1. SETUP & DATA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "Suicide_Severity_Final.csv"
MODEL_CHECKPOINT = "microsoft/deberta-v3-small"

df = pd.read_csv(DATA_PATH)
df['label'] = df['severity'].astype(int)
df = df.dropna(subset=['text'])

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.15) # 15% for validation

# 2. PREPROCESSING
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. METRICS
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 4. TRAINING
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=6
).to(DEVICE)

training_args = TrainingArguments(
    output_dir="severity_model_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Increase if you have 12GB+ VRAM
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

print("[INFO] Starting Training on AMD GPU...")
trainer.train()

# 5. SAVE FINAL MODEL
model.save_pretrained("./final_severity_model")
tokenizer.save_pretrained("./final_severity_model")
print("[INFO] Model saved to ./final_severity_model")