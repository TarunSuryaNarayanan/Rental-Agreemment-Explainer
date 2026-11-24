# rae/train_risk_model.py
"""
Fine‑tune a tiny transformer to classify rental‑agreement clauses
into risk levels: Standard, Low, Medium, High, Critical.

Usage:
    python -m rae.train_risk_model \
          --train-path ./data/risk_training.csv \
          --model-base google/gemma-2b-it \
          --output-dir ./models/risk-clf \
          --epochs 3 \
          --batch-size 8 \
          --lr 5e-5
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --------------------------------------------------------------
# 1️⃣  Dataset
# --------------------------------------------------------------
class ClauseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512):
        self.texts = df["clause_text"].tolist()
        self.labels = df["risk_label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        # map label string -> integer
        self.label2id = {lbl: i for i, lbl in enumerate(["Standard", "Low", "Medium", "High", "Critical"])}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        lbl = self.label2id[self.labels[idx]]

        # Prompt‑style input – the model was trained on instruction data
        prompt = f"Classify the risk level of the following clause. Return ONLY the label.\nClause: {txt}"
        enc = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # The target is the label token (e.g., "High")
        target = self.tokenizer(
            self.id2label[lbl],
            max_length=4,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Flatten tensors (remove batch dim)
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        target = {k: v.squeeze(0) for k, v in target.items()}
        target["labels"] = target["input_ids"]  # seq2seq expects `labels`

        return {**enc, **target}

# --------------------------------------------------------------
# 2️⃣  Training harness
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine‑tune risk‑classification model")
    parser.add_argument("--train-path", type=str, required=True, help="Path to CSV with clause_text,risk_label")
    parser.add_argument("--model-base", type=str, default="google/gemma-2b-it", help="Base transformer checkpoint")
    parser.add_argument("--output-dir", type=str, default="./models/risk-clf", help="Where to save fine‑tuned model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    # 2️⃣ Load data
    df = pd.read_csv(args.train_path)
    df = df.dropna(subset=["clause_text", "risk_label"])
    df["risk_label"] = df["risk_label"].str.strip()

    # 3️⃣ Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_base)

    # 4️⃣ Dataset & Trainer
    dataset = ClauseDataset(df, tokenizer, max_len=args.max_len)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("=== Starting fine‑tuning ===")
    trainer.train()
    print("=== Saving fine‑tuned model ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
