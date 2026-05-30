"""Flask web dashboard + REST API for real-time phishing detection."""
import os
import time
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from feature_extraction import extract_all, FEATURE_NAMES

MODELS_DIR = os.environ.get('MODELS_DIR', 'models')

app = Flask(__name__, template_folder='../templates')
_model = None
_scaler = None
_explainer = None


def load():
    global _model, _scaler
    if _model is None:
        _model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'phishing_dnn.keras'))
        _scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    return _model, _scaler


def get_explainer():
    global _explainer
    if _explainer is None:
        from explain import Explainer
        _explainer = Explainer(MODELS_DIR)
    return _explainer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    url = data.get('url', '').strip()
    want_explain = bool(data.get('explain', False))
    if not url:
        return jsonify({'error': 'url required'}), 400

    model, scaler = load()
    t0 = time.time()
    feats = extract_all(url)
    x = scaler.transform(feats.reshape(1, -1))
    p = float(model.predict(x, verbose=0)[0, 0])
    elapsed_ms = (time.time() - t0) * 1000

    result = {
        'url': url,
        'probability': p,
        'label': 'phishing' if p >= 0.5 else 'legitimate',
        'confidence': p if p >= 0.5 else 1 - p,
        'response_ms': round(elapsed_ms, 2),
    }
    if want_explain:
        result['top_features'] = [
            {'feature': f, 'shap': s, 'value': v}
            for f, s, v in get_explainer().explain_url(url, top_k=8)
        ]
    return jsonify(result)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'features': len(FEATURE_NAMES)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
