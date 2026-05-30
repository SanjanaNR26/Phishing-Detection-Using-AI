"""Evaluate trained DNN: confusion matrix, ROC, classification report."""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
import tensorflow as tf

from preprocess import build_dataset, prepare


def evaluate(data_csv, models_dir='models'):
    model = tf.keras.models.load_model(os.path.join(models_dir, 'phishing_dnn.keras'))
    X, y = build_dataset(data_csv)
    _, _, (Xte, yte) = prepare(X, y, models_dir)

    proba = model.predict(Xte).ravel()
    pred = (proba >= 0.5).astype(int)

    print("\n=== Classification report ===")
    print(classification_report(yte, pred, target_names=['Legitimate', 'Phishing'], digits=4))

    cm = confusion_matrix(yte, pred)
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.title('Confusion Matrix — TensorFlow DNN')
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'), dpi=120)

    fpr, tpr, _ = roc_curve(yte, proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'DNN (AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend()
    plt.savefig(os.path.join(models_dir, 'roc_curve.png'), dpi=120)
    print("Plots saved to", models_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    evaluate(args.data)
