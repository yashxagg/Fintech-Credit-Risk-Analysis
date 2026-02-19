  # 💳 Fintech Credit Risk Engine

### **End-to-End Loan Default Prediction & Risk Analytics Portal**

![Python Version](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)

## 📌 Project Overview
This project addresses the critical challenge of credit risk in the Fintech sector. By analyzing historical lending data, I developed a machine learning model that identifies high-risk applicants, helping financial institutions make data-driven lending decisions.

Unlike static notebooks, this is a **production-ready tool** featuring an interactive "Credit Officer Portal" built entirely in Python. It allows for real-time risk assessment and "What-If" scenario analysis for individual loan applications.



---

## 🛠️ Tech Stack & Skills

* **Machine Learning:** XGBoost, Random Forest, Scikit-Learn.
* **Data Engineering:** Python (Pandas, Numpy), SQL (Window Functions & CTEs for risk segmentation).
* **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique).
* **Dashboard & Deployment:** Streamlit (Custom UI & Real-time Visuals).
* **Visualization:** Matplotlib, Seaborn (Integrated into Streamlit).

---

## 🚀 Key Features

### 1. **Automated Risk Scoring**
Real-time classification of "Default" vs. "Paid" status. The model outputs a probability score, which is translated into a **Dynamic Risk Meter** (Low, Medium, High).

### 2. **Feature Importance Analysis**
Uses model transparency to identify the top drivers of risk. 
* **Key Finding:** Debt-to-Income (DTI) ratios above 30% were identified as the strongest predictor of default.

### 3. **Interactive "What-If" Simulator**
A dedicated interface where credit officers can adjust applicant parameters (Income, Loan Amount, Credit History) to see live changes in risk probability before approving a loan.

---

## 📈 Business Impact & Evaluation

### **Metric Focus: Recall**
In lending, a **False Negative** (missing a defaulter) is significantly more expensive than a **False Positive** (rejecting a good applicant). Therefore, this model is optimized for **Recall** to protect the lender's capital.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 96% |
| **Recall (Defaulters)** | 98% |
| **Precision** | 94% |



---

## 📂 Project Structure
```bash
├── data/               # Historical lending datasets
├── notebooks/          # EDA and Model Training (Jupyter)
├── src/                # SQL queries and Python processing scripts
├── model/              # Serialized .pkl files
├── app.py              # Streamlit Application
└── requirements.txt    # Dependencies
```
---


## ⚙️ Installation & Usage
### 1. Clone the repository.
```bash
git clone https://github.com/yashxagg/Fintech-Credit-Risk-Engine.git
cd Fintech-Credit-Risk-Engine
```
### 2. Install dependencies.
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App:
```bash
streamlit run app.py
```

---

## 👤 Author
* **Yash Aggarwal**
  * 🎓 B.Tech CSE (AI & ML) | Class of 2026
  * 🐙 [GitHub Profile](https://github.com/yashxagg)
  * 💼 [LinkedIn Profile](https://linkedin.com/in/yash-aggarwal0812)
