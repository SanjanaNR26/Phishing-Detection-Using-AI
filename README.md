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
│   ├── Command-line.py         # Command-line detector
│   └── app.py                  # Flask web dashboard + REST API
├── templates                   # Dashboard UI
├── data                        # Place PhishTank/UCI/OpenPhish CSVs here
├── models                      # Saved .keras model + scaler.pkl
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

# 3. Command-line inference
python src/Command-line.py "http://paypa1-secure-login.com/verify"

# 4. Web dashboard
python src/app.py        # http://localhost:5000
```

## Datasets
- PhishTank: https://www.phishtank.com/
- OpenPhish: https://openphish.com/
- UCI Phishing Websites: https://archive.ics.uci.edu/ml/datasets/phishing+websites
- ISCX-URL-2016

CSV format expected: `url,label` (label: 1 = phishing, 0 = legitimate).


## Requirements
```bash
tensorflow==2.12.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
nltk==3.8.1
shap==0.44.0
flask==2.3.3
matplotlib==3.8.2
seaborn==0.13.0
imbalanced-learn==0.11.0
joblib==1.3.2
python-whois==0.8.0
tldextract==5.1.1
requests==2.31.0
```
