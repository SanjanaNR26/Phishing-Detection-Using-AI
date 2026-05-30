"""Baseline classifier comparison: RF, SVM, LogReg, Naive Bayes."""
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from preprocess import build_dataset, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()

    X, y = build_dataset(args.data)
    (Xtr, ytr), _, (Xte, yte) = prepare(X, y)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        'SVM (RBF)': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
    }

    rows = []
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(yte, pred),
            'Precision': precision_score(yte, pred),
            'Recall': recall_score(yte, pred),
            'F1': f1_score(yte, pred),
            'AUC-ROC': roc_auc_score(yte, proba),
        })

    df = pd.DataFrame(rows)
    print("\n=== Baseline comparison ===")
    print(df.to_string(index=False))
    df.to_csv('models/baseline_results.csv', index=False)


if __name__ == '__main__':
    main()
