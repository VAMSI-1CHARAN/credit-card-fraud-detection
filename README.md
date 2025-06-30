# Credit Card Fraud Detection 🚨💳

This project focuses on detecting fraudulent credit card transactions using machine learning. It uses the publicly available **Kaggle dataset**, which contains transactions made by European cardholders over two days in 2013.

---

## 📊 Dataset

* **Source**: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Size**: 284,807 transactions with 492 frauds (highly imbalanced)
* **Features**: V1–V28 (PCA-transformed), Time, Amount, and Class (fraud = 1, non-fraud = 0)

---

## 🔍 Problem Statement

Build a classification model to accurately detect fraudulent transactions in highly imbalanced data using techniques such as:

* Logistic Regression
* SMOTE (Synthetic Minority Oversampling Technique)
* Evaluation metrics like precision, recall, F1-score

---

## 🧪 Project Structure

```
credit-card-fraud-detection/
├── fraud_analysis.py            # Model training & evaluation
├── fraud_visualizations.py     # Visualizations of fraud vs non-fraud
├── fraud_detection_results.csv # Final model predictions
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repo**:

```bash
git clone https://github.com/VAMSI-1CHARAN/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create and activate a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the analysis script**:

```bash
python3 fraud_analysis.py
```

5. *(Optional)* Run visualizations:

```bash
python3 fraud_visualizations.py
```

---

## 📈 Visualizations

* Class distribution (fraud vs non-fraud)
* Transaction amount vs fraud likelihood
* Correlation heatmap of features
* Model performance metrics

---

## ✅ Final Notes

* The model is evaluated **before and after using SMOTE** to highlight the impact of handling class imbalance.
* You can modify the model (e.g., try Random Forest, XGBoost, etc.) and add performance comparisons.

---

## 📬 Contact

Feel free to connect on [LinkedIn](https://www.linkedin.com/) or raise an issue if you'd like to collaborate.
