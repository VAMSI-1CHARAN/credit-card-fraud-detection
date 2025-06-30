# 🛡️ Credit Card Fraud Detection - Machine Learning Project

## 📌 Overview

This project focuses on identifying fraudulent credit card transactions using Machine Learning techniques. Given the severe class imbalance in real-world fraud datasets, this project demonstrates preprocessing, modeling (with and without SMOTE), evaluation, and visualizations of transaction data to effectively detect fraud.

---

## 🧠 Problem Statement

Credit card fraud is a major concern for banks and financial institutions. Only a tiny fraction of all transactions are fraudulent, making it hard to detect them accurately. The challenge lies in:

* Handling class imbalance
* Achieving high precision and recall on minority class (fraud)
* Preventing false positives that affect legitimate customers

---

## 📂 Dataset

* **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Size:** \~284,807 transactions, 492 frauds
* **Features:**

  * 28 PCA-transformed features: `V1` to `V28`
  * `Time` and `Amount` (scaled)
  * `Class` (0 = non-fraud, 1 = fraud)

---

## ⚙️ Technologies Used

* Python 3.13
* Pandas, NumPy, Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib, Seaborn

---

## ✅ What We Have Done

* ✅ Cleaned and explored the dataset
* ✅ Scaled `Time` and `Amount` features
* ✅ Split data into training and test sets with class balance preserved
* ✅ Built a baseline Logistic Regression model
* ✅ Applied SMOTE to handle class imbalance
* ✅ Evaluated pre- and post-SMOTE models using confusion matrix & classification report
* ✅ Visualized class distribution, feature correlations, and fraud patterns

---

## 📊 Visualizations

* Fraud vs Non-Fraud class distribution
* Correlation heatmaps of features
* Distribution of transaction amount in fraud vs non-fraud
* Time-based fraud activity patterns

---

## 🚀 Use Cases & Implementation Scenarios

### 1. 🏦 **Banking Systems**

* Flag suspicious transactions in real-time
* Integrated into internal fraud detection tools

### 2. 💳 **Payment Gateways**

* Stripe, Razorpay, CCAvenue — to detect card-not-present fraud

### 3. 📱 **FinTech Apps**

* Apps like Google Pay, PhonePe, Paytm use such ML pipelines to minimize risk

### 4. 🛒 **E-commerce Platforms**

* Detect fake orders or identity theft

### 5. 🔐 **Cybersecurity Tools**

* Feed into SIEM systems to raise automated fraud alerts

### 6. 📈 **Data Analytics Teams**

* Analyze fraud trends and update risk rules accordingly

---

## 🔗 Future Work

* Deploy the model via Flask API for real-time use
* Use advanced models like XGBoost, LightGBM, or Isolation Forest
* Monitor model drift and re-train periodically
* Integrate with a dashboard (Power BI / Streamlit)

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
├── creditcard.csv                  # Dataset (locally stored, not pushed due to size >100MB)
├── fraud_analysis.py              # ML logic and model evaluation
├── fraud_visualizations.py        # Visual analysis and plots
├── fraud_detection_results.csv    # Post-SMOTE results
└── README.md                      # Project documentation
```

---

## ⚠️ Note

Due to GitHub's 100MB file limit, the dataset is not uploaded. Download it manually from Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🙋‍♂️ Author

**Vamsi Charan**

* GitHub: [VAMSI-1CHARAN](https://github.com/VAMSI-1CHARAN)

---

Feel free to ⭐️ this repo and share your feedback!
