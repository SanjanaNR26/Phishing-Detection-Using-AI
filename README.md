# Phishing-Detection-Using-AI

A deep-learning phishing detector built with **Python + TensorFlow**, served via a
**Flask** web dashboard and CLI, with **SHAP** explainability.

## Project Structure
```
phishing_detector/
├── src/
│   ├── feature_extraction.py   # URL + email feature engineering (48 features)
│   ├── preprocess.py           # Load datasets, balance, scale, split
│   ├── model.py                # TensorFlow DNN architecture
│   ├── train.py                # Training pipeline (Adam, dropout, early stop)
│   ├── evaluate.py             # Accuracy / precision / recall / ROC / confusion
│   ├── baselines.py            # RF, SVM, LogReg, NB comparison
│   ├── explain.py              # SHAP explanations
│   ├── cli.py                  # Command-line detector
│   └── app.py                  # Flask web dashboard + REST API
├── templates/index.html        # Dashboard UI
├── data/                       # Place PhishTank/UCI/OpenPhish CSVs here
├── models/                     # Saved .keras model + scaler.pkl
└── requirements.txt
```

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# 1. Train
python src/train.py --data data/urls.csv --epochs 50

# 2. Evaluate + compare baselines
python src/evaluate.py
python src/baselines.py

# 3. CLI inference
python src/cli.py "http://paypa1-secure-login.com/verify"

# 4. Web dashboard
python src/app.py        # http://localhost:5000
```

## Datasets
- PhishTank: https://www.phishtank.com/
- OpenPhish: https://openphish.com/
- UCI Phishing Websites: https://archive.ics.uci.edu/ml/datasets/phishing+websites
- ISCX-URL-2016

CSV format expected: `url,label` (label: 1 = phishing, 0 = legitimate).


<img width="1919" height="1017" alt="Screenshot 2026-05-30 135405" src="https://github.com/user-attachments/assets/4549880c-96ae-43fa-8a25-76838a5f6d10" />
