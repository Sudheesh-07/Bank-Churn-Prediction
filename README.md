# Bank Churn Prediction

This project predicts whether a customer is likely to leave (churn) a bank based on their details using a machine learning model.

## 📁 Files

* `BankChurners.csv` – Dataset used for training
* `churn_prediction_model.pkl` – Trained model file
* `app_final_model.py` – Main Flask app
* `app_streamlit.py` – Streamlit version of the app
* `1_Customer_Analysis.py` – Data analysis script
* `templates/` – HTML files for the web app
* `requirements.txt` – List of Python libraries used

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

#### Flask version:

```bash
python app_final_model.py
```


#### Streamlit version:

```bash
streamlit run app_streamlit.py
```

## ⚙️ Features

* Predicts if a customer will churn
* Web interface using Flask and Streamlit
* Simple input form to enter customer details
* Uses Random Forest model for prediction

## 📊 Dataset Columns

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

## ✅ Output

* Churn or Not Churn
* Probability Score
