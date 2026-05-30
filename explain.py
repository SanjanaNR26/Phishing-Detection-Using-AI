"""SHAP-based explainability for the DNN."""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

from feature_extraction import extract_all, FEATURE_NAMES


class Explainer:
    def __init__(self, models_dir='models', background=None):
        self.model = tf.keras.models.load_model(os.path.join(models_dir, 'phishing_dnn.keras'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        if background is None:
            background = np.zeros((50, len(FEATURE_NAMES)), dtype=np.float32)
        self.explainer = shap.KernelExplainer(
            lambda x: self.model.predict(x, verbose=0).ravel(),
            background,
        )

    def explain_url(self, url, top_k=10):
        x = self.scaler.transform(extract_all(url).reshape(1, -1))
        sv = self.explainer.shap_values(x, nsamples=100)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).ravel()
        order = np.argsort(np.abs(sv))[::-1][:top_k]
        return [(FEATURE_NAMES[i], float(sv[i]), float(x[0, i])) for i in order]


def plot_global_importance(X_sample, models_dir='models'):
    exp = Explainer(models_dir, background=X_sample[:100])
    sv = exp.explainer.shap_values(X_sample[:200], nsamples=100)
    if isinstance(sv, list):
        sv = sv[0]
    shap.summary_plot(sv, X_sample[:200], feature_names=FEATURE_NAMES,
                      plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'shap_importance.png'), dpi=120)


if __name__ == '__main__':
    import sys
    e = Explainer()
    for f, s, v in e.explain_url(sys.argv[1] if len(sys.argv) > 1 else "http://paypa1-login.xyz/verify"):
        print(f"{f:30s}  shap={s:+.4f}  value={v:.3f}")
