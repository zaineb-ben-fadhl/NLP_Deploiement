import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import mlflow

from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers import DebertaV2Tokenizer

# ======================================================
# 0) MLflow
# ======================================================
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("dreaddit-deberta-v3")

# ======================================================
# 1) Paths
# ======================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dreaddit")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
RESULTS_DIR = os.path.join(RUNS_DIR, "results_deberta")
LOGS_DIR = os.path.join(RUNS_DIR, "logs_deberta")

def ensure_dirs():
    for d in [
        DATA_DIR, ARTIFACTS_DIR, MODELS_DIR,
        METRICS_DIR, PLOTS_DIR, REPORTS_DIR,
        RUNS_DIR, RESULTS_DIR, LOGS_DIR
    ]:
        os.makedirs(d, exist_ok=True)

# ======================================================
# 2) Config
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "microsoft/deberta-v3-large"

MAX_LENGTH = 128
DROPOUT_PROB = 0.2
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

AUGMENT_RATIO = 0.25
RANDOM_STATE = 42

MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "deberta_dreaddit_best")

# ======================================================
# 3) Dataset cache
# ======================================================
def get_dataset_cached():
    if os.path.exists(DATASET_DIR):
        print(f"Loading cached dataset from: {DATASET_DIR}")
        return load_from_disk(DATASET_DIR)

    print("Downloading dataset: andreagasparini/dreaddit")
    ds = load_dataset("andreagasparini/dreaddit")
    ds.save_to_disk(DATASET_DIR)
    return ds

# ======================================================
# 4) Data preparation
# ======================================================
def load_and_prepare_data(random_state=42):
    ds = get_dataset_cached()
    df = pd.concat(
        [pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])],
        ignore_index=True
    )

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["clean_text"] = df["text"].apply(clean_text)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=random_state
    )

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df

def augment_train_df(train_df, augment_ratio=0.25, random_state=42):
    rng = np.random.RandomState(random_state)

    def augment_text(text):
        words = text.split()
        if len(words) < 6:
            return text

        if rng.rand() < 0.05:
            words.pop(rng.randint(len(words)))

        if rng.rand() < 0.05 and len(words) > 1:
            i = rng.randint(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]

        return " ".join(words)

    n_aug = int(len(train_df) * augment_ratio)
    aug_df = train_df.sample(n=n_aug, random_state=random_state).copy()
    aug_df["clean_text"] = aug_df["clean_text"].apply(augment_text)

    train_aug = pd.concat([train_df, aug_df], ignore_index=True)
    print("Train après augmentation:", len(train_aug))
    return train_aug

# ======================================================
# 5) Torch Dataset
# ======================================================
class StressDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def build_dataset(tokenizer, df):
    encodings = tokenizer(
        df["clean_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    return StressDataset(encodings, df["label"].tolist())

# ======================================================
# 6) Metrics
# ======================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# ======================================================
# 7) Model init
# ======================================================
def model_init():
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)

    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = DROPOUT_PROB
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = DROPOUT_PROB
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = DROPOUT_PROB

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config
    ).to(device)

# ======================================================
# 8) Confusion matrix
# ======================================================
def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ======================================================
# 9) Main
# ======================================================
def main():
    ensure_dirs()

    train_df, val_df, test_df = load_and_prepare_data(RANDOM_STATE)
    train_df = augment_train_df(train_df, AUGMENT_RATIO, RANDOM_STATE)

    # 🔥 IMPORTANT FIX
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )

    train_dataset = build_dataset(tokenizer, train_df)
    val_dataset = build_dataset(tokenizer, val_df)
    test_dataset = build_dataset(tokenizer, test_df)

    with mlflow.start_run(run_name="deberta-v3-large"):

        training_args = TrainingArguments(
            output_dir=RESULTS_DIR,
            num_train_epochs=MAX_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            logging_dir=LOGS_DIR,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=[]
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )],
        )

        trainer.train()

        test_output = trainer.predict(test_dataset)
        y_true = test_output.label_ids
        y_pred = np.argmax(test_output.predictions, axis=-1)

        trainer.save_model(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)

        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_true, y_pred, cm_path)

        report = classification_report(y_true, y_pred, digits=4)
        report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifacts(MODEL_SAVE_DIR, artifact_path="model")

        print("✅ Training terminé avec succès")

if __name__ == "__main__":
    main()
