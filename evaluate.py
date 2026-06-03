import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from data import make_gens

DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "best_model.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
    }


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_markdown_table(auc_val: float, r30: dict, r50: dict, path: str):
    md = []
    md.append("## Threshold Trade-off (Test Set)\n")
    md.append(f"ROC-AUC (threshold-independent): **{auc_val:.3f}**\n")
    md.append("| Mode | Threshold | Accuracy | Precision | Recall | F1 |")
    md.append("|------|-----------|----------|-----------|--------|----|")
    md.append(
        f"| High Recall | 0.30 | {r30['accuracy']:.3f} | {r30['precision']:.3f} | {r30['recall']:.3f} | {r30['f1']:.3f} |"
    )
    md.append(
        f"| Balanced | 0.50 | {r50['accuracy']:.3f} | {r50['precision']:.3f} | {r50['recall']:.3f} | {r50['f1']:.3f} |"
    )
    md.append("\n**Trade-off:** Lower threshold (0.30) increases recall (fewer missed pneumonia cases) but may increase false positives.")
    md.append("Higher threshold (0.50) is more conservative, improving precision while risking more false negatives.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def plot_and_save_confusion_matrix(y_true, y_prob, thr: float, class_names, out_path: str):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (thr={thr:.2f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_roc(y_true, y_prob, out_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

_, _, test_gen, class_names = make_gens(DATA_DIR, IMG_SIZE, BATCH_SIZE, val_split=0.1)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
y_prob = model.predict(test_gen, verbose=1).ravel()
y_true = test_gen.classes

auc_val = float(roc_auc_score(y_true, y_prob))

r30 = metrics_at_threshold(y_true, y_prob, 0.30)
r50 = metrics_at_threshold(y_true, y_prob, 0.50)

# save metrics
metrics_payload = {
    "model_path": MODEL_PATH,
    "roc_auc": auc_val,
    "threshold_0_30": r30,
    "threshold_0_50": r50,
}
save_json(metrics_payload, os.path.join(OUTPUT_DIR, "metrics.json"))

# save plots
plot_and_save_confusion_matrix(
    y_true, y_prob, 0.30, class_names,
    os.path.join(OUTPUT_DIR, "cm_thr_0.30.png")
)
plot_and_save_confusion_matrix(
    y_true, y_prob, 0.50, class_names,
    os.path.join(OUTPUT_DIR, "cm_thr_0.50.png")
)
plot_and_save_roc(
    y_true, y_prob,
    os.path.join(OUTPUT_DIR, "roc_curve.png")
)