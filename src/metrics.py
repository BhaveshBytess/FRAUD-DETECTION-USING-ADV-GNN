# src/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support

def compute_metrics(y_true, y_prob, thresh=0.5):
    # Ensure y_true is not all the same value, which would break AUC calculation
    if len(np.unique(y_true)) < 2:
        return {'auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'error': 'Only one class present in y_true.'}

    try:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError as e:
        return {'auc': 0.0, 'pr_auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'error': str(e)}

    y_pred = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return {'auc': float(auc), 'pr_auc': float(pr_auc), 'f1': float(f1), 'precision': float(p), 'recall': float(r)}
