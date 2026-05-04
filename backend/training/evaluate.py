"""
Evaluation utilities — run against the test split and print a full report.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_dir: Path = None,
) -> dict:
    """
    Run inference on loader, compute all metrics, print a report.
    Optionally saves results to results_dir/evaluation.json.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits.float(), dim=1)
            preds  = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    metrics = {}

    if HAS_SKLEARN:
        metrics['accuracy']  = round(accuracy_score(all_labels, all_preds) * 100, 2)
        metrics['f1_score']  = round(f1_score(all_labels, all_preds, average='binary', zero_division=0), 4)
        try:
            metrics['auc_roc'] = round(roc_auc_score(all_labels, all_probs), 4)
        except Exception:
            metrics['auc_roc'] = None

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        metrics['precision'] = round(tp / max(tp + fp, 1), 4)
        metrics['recall']    = round(tp / max(tp + fn, 1), 4)

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy  : {metrics['accuracy']:.1f}%")
        print(f"F1 Score  : {metrics['f1_score']:.4f}")
        print(f"AUC-ROC   : {metrics.get('auc_roc', 'N/A')}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={tn}  FP={fp}")
        print(f"  FN={fn}  TP={tp}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=['Real', 'Fake'], zero_division=0))
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metrics['accuracy'] = round(correct / len(all_labels) * 100, 2)
        print(f"Accuracy: {metrics['accuracy']:.1f}%  (install scikit-learn for full metrics)")

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out = Path(results_dir) / "evaluation.json"
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {out}")

    return metrics
