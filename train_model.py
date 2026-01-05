import os
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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

import mlflow
import matplotlib.pyplot as plt


# -----------------------------
# 0) MLflow config (workshop-style)
# -----------------------------
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("dreaddit-deberta-prediction")


# -----------------------------
# 1) Paths (MLOps-friendly)
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dreaddit")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")

RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
RESULTS_DIR = os.path.join(RUNS_DIR, "results_deberta_final")
LOGS_DIR = os.path.join(RUNS_DIR, "logs_deberta_final")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


# -----------------------------
# 2) Config (same logic as yours)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "microsoft/deberta-v3-large"
DROPOUT_PROB = 0.2
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
AUGMENT_RATIO = 0.25
RANDOM_STATE = 42
MAX_LENGTH = 128

MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "deberta_dreaddit_best")
METRICS_JSON_PATH = os.path.join(METRICS_DIR, "final_metrics_deberta.json")
METRICS_CSV_PATH = os.path.join(METRICS_DIR, "final_metrics_deberta.csv")


# -----------------------------
# 3) Dataset cache (download once)
# -----------------------------
def get_dataset_cached():
    if os.path.exists(DATASET_DIR):
        print(f"✅ Loading cached dataset from: {DATASET_DIR}")
        return load_from_disk(DATASET_DIR)

    print("⬇️ Downloading dataset: andreagasparini/dreaddit")
    ds = load_dataset("andreagasparini/dreaddit")
    print(f"💾 Saving dataset to disk: {DATASET_DIR}")
    ds.save_to_disk(DATASET_DIR)
    return ds


# -----------------------------
# 4) Data prep (same logic)
# -----------------------------
def load_and_prepare_data(random_state=42):
    ds = get_dataset_cached()
    df_all = pd.concat(
        [pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])],
        ignore_index=True,
    )

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df_all["clean_text"] = df_all["text"].apply(clean_text)

    train_df, temp_df = train_test_split(
        df_all, test_size=0.2, stratify=df_all["label"], random_state=random_state
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

    def augment_text_light(text: str) -> str:
        words = text.split()
        if len(words) < 6:
            return text

        if rng.rand() < 0.05 and len(words) > 6:
            words.pop(rng.randint(0, len(words)))

        if rng.rand() < 0.05 and len(words) >= 8:
            i = rng.randint(0, len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]

        return " ".join(words)

    n_aug = int(len(train_df) * augment_ratio)
    aug_df = train_df.sample(n=n_aug, random_state=random_state).copy()
    aug_df["clean_text"] = aug_df["clean_text"].apply(augment_text_light)

    train_aug = pd.concat([train_df, aug_df], ignore_index=True)
    print("Train après augmentation:", len(train_aug))
    return train_aug


# -----------------------------
# 5) Torch dataset
# -----------------------------
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
    enc = tokenizer(
        df["clean_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    return StressDataset(enc, df["label"].tolist())


# -----------------------------
# 6) Metrics (same as yours)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


# -----------------------------
# 7) Model init (dropout=0.2)
# -----------------------------
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
        config=config,
    ).to(device)


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ensure_dirs()

    print("Chargement + preparation des donnees...")
    train_df, val_df, test_df = load_and_prepare_data(random_state=RANDOM_STATE)

    print("\nAugmentation TRAIN uniquement...")
    train_df = augment_train_df(
        train_df, augment_ratio=AUGMENT_RATIO, random_state=RANDOM_STATE
    )

    # ✅ FIX IMPORTANT: avoid DeBERTa fast-tokenizer conversion bug
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_dataset = build_dataset(tokenizer, train_df)
    val_dataset = build_dataset(tokenizer, val_df)
    test_dataset = build_dataset(tokenizer, test_df)

    print("\nEntrainement du modele...")
    with mlflow.start_run(run_name="deberta-v3-large-v1"):

        params = {
            "model_name": MODEL_NAME,
            "dropout_prob": DROPOUT_PROB,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "augment_ratio_train_only": AUGMENT_RATIO,
            "random_state": RANDOM_STATE,
            "max_length": MAX_LENGTH,
        }
        mlflow.log_params(params)

        training_args = TrainingArguments(
            output_dir=RESULTS_DIR,
            num_train_epochs=MAX_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            logging_dir=LOGS_DIR,
            logging_steps=50,
            report_to=[],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
            ],
        )

        trainer.train()

        train_metrics = trainer.evaluate(train_dataset)
        val_metrics = trainer.evaluate(val_dataset)
        test_metrics = trainer.evaluate(test_dataset)

        mlflow.log_metrics({
            "train_loss": float(train_metrics.get("eval_loss", 0.0)),
            "train_accuracy": float(train_metrics.get("eval_accuracy", 0.0)),
            "train_f1": float(train_metrics.get("eval_f1", 0.0)),
            "train_precision": float(train_metrics.get("eval_precision", 0.0)),
            "train_recall": float(train_metrics.get("eval_recall", 0.0)),

            "val_loss": float(val_metrics.get("eval_loss", 0.0)),
            "val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
            "val_f1": float(val_metrics.get("eval_f1", 0.0)),
            "val_precision": float(val_metrics.get("eval_precision", 0.0)),
            "val_recall": float(val_metrics.get("eval_recall", 0.0)),

            "test_loss": float(test_metrics.get("eval_loss", 0.0)),
            "test_accuracy": float(test_metrics.get("eval_accuracy", 0.0)),
            "test_f1": float(test_metrics.get("eval_f1", 0.0)),
            "test_precision": float(test_metrics.get("eval_precision", 0.0)),
            "test_recall": float(test_metrics.get("eval_recall", 0.0)),
        })

        final_metrics = {
            "model": MODEL_NAME,
            "date": datetime.now().isoformat(),
            "dropout": DROPOUT_PROB,
            "hyperparams": params,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }

        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=4)

        rows = []
        for split_name, split_metrics in [
            ("train", train_metrics),
            ("validation", val_metrics),
            ("test", test_metrics),
        ]:
            row = {"split": split_name}
            row.update(split_metrics)
            rows.append(row)
        pd.DataFrame(rows).to_csv(METRICS_CSV_PATH, index=False)

        mlflow.log_artifact(METRICS_JSON_PATH)
        mlflow.log_artifact(METRICS_CSV_PATH)

        preds_output = trainer.predict(test_dataset)
        y_true = preds_output.label_ids
        y_pred = np.argmax(preds_output.predictions, axis=-1)

        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        save_confusion_matrix(y_true, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        report_txt = classification_report(y_true, y_pred, digits=4)
        report_path = os.path.join(REPORTS_DIR, "classification_report_test.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        mlflow.log_artifact(report_path)

        trainer.save_model(MODEL_SAVE_DIR)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        mlflow.log_artifacts(MODEL_SAVE_DIR, artifact_path="model_artifacts")

        mlflow.set_tags({
            "environment": "development",
            "model_type": "DeBERTa",
            "task": "binary_classification",
            "dataset": "andreagasparini/dreaddit",
        })

        print("\n" + "=" * 60)
        print("RESULTATS DE L'ENTRAINEMENT (DeBERTa Dreaddit)")
        print("=" * 60)
        print(f"Val F1       : {val_metrics.get('eval_f1', 0.0):.4f}")
        print(f"Test F1      : {test_metrics.get('eval_f1', 0.0):.4f}")
        print(f"Test Accuracy: {test_metrics.get('eval_accuracy', 0.0):.4f}")
        print("=" * 60)

        print(f"\nModele sauvegarde localement dans : {MODEL_SAVE_DIR}")
        print(f"Metrics JSON : {METRICS_JSON_PATH}")
        print(f"Metrics CSV  : {METRICS_CSV_PATH}")
        print("\nMLflow UI    : mlflow ui --port 5000")
        print("=" * 60)


if __name__ == "__main__":
    main()
