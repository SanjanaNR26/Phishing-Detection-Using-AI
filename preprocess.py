"""Load CSV, extract features, balance, scale, split."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from feature_extraction import extract_all, FEATURE_NAMES


def build_dataset(csv_path, enable_network=False, max_rows=None):
    df = pd.read_csv(csv_path)
    assert 'url' in df.columns and 'label' in df.columns, "CSV needs columns: url,label"
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    print(f"Extracting features from {len(df)} URLs...")
    X = np.vstack([extract_all(u, enable_network=enable_network) for u in df['url']])
    y = df['label'].astype(int).values
    return X, y


def prepare(X, y, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    # Stratified split: 70/15/15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # SMOTE on train only
    sm = SMOTE(random_state=42)
    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(FEATURE_NAMES, os.path.join(models_dir, 'feature_names.pkl'))
    print(f"Train {X_tr.shape}  Val {X_val.shape}  Test {X_te.shape}")
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='data/processed.npz')
    args = p.parse_args()
    X, y = build_dataset(args.data)
    np.savez(args.out, X=X, y=y)
    print("Saved", args.out)
