# 🛡️ CreditGuard AI
**Credit Card Fraud Detection App**

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to use
1. Open **http://localhost:8501**
2. Click **"Try Demo Data"** to test instantly, OR upload your own CSV
3. CreditGuard AI scans every transaction and flags fraud in seconds
4. Download the report

## CSV Format
Your file should have columns like:
```
Time, V1, V2, ..., V28, Amount
```
(Compatible with Kaggle Credit Card Fraud dataset)
