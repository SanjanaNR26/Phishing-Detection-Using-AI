"""Command-line phishing detector."""
import argparse
import os
import sys
import joblib
import numpy as np
import tensorflow as tf

from feature_extraction import extract_all


def predict(url, models_dir='models'):
    model = tf.keras.models.load_model(os.path.join(models_dir, 'phishing_dnn.keras'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    x = scaler.transform(extract_all(url).reshape(1, -1))
    p = float(model.predict(x, verbose=0)[0, 0])
    return p


def main():
    ap = argparse.ArgumentParser(description='AI Phishing URL Detector')
    ap.add_argument('url', nargs='?', help='URL to classify')
    ap.add_argument('--file', help='File with one URL per line')
    ap.add_argument('--explain', action='store_true')
    args = ap.parse_args()

    urls = []
    if args.url:
        urls.append(args.url)
    if args.file:
        urls += [l.strip() for l in open(args.file) if l.strip()]
    if not urls:
        ap.print_help(); sys.exit(1)

    for u in urls:
        p = predict(u)
        label = 'PHISHING' if p >= 0.5 else 'LEGITIMATE'
        print(f"[{label}]  p={p:.4f}  {u}")
        if args.explain:
            from explain import Explainer
            for f, s, v in Explainer().explain_url(u, top_k=5):
                print(f"   {f:28s} shap={s:+.4f}")


if __name__ == '__main__':
    main()
