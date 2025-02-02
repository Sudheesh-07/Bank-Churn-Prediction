import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1, recall_score as recall
from imblearn.over_sampling import SMOTE
from plotly.subplots import make_subplots

# Streamlit Page Config
st.set_page_config(layout="wide", page_title="Bank Churn Analysis")

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("BankChurners.csv")
        data = data[data.columns[:-2]]  # Drop last two columns
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

# Load Data
c_data = load_data()

# Title
st.title("📊 Bank Customer Churn Analysis")

# Show dataset preview if checked
if not c_data.empty and st.checkbox("Show raw data"):
    st.write(c_data.head())

# --- 📌 CHART 1: Age Distribution ---
st.header("📈 Distribution of Customer Ages")

if "Customer_Age" in c_data.columns:
    fig1 = make_subplots(rows=2, cols=1)

    # Box Plot for Age
    tr1 = go.Box(x=c_data['Customer_Age'], name='Age Box Plot', boxmean=True)
    
    # Histogram for Age
    tr2 = go.Histogram(x=c_data['Customer_Age'], name='Age Histogram')

    # Add traces to figure
    fig1.add_trace(tr1, row=1, col=1)
    fig1.add_trace(tr2, row=2, col=1)

    # Update layout
    fig1.update_layout(height=700, width=1200, title_text="Distribution of Customer Ages")

    # Display plot in Streamlit
    st.plotly_chart(fig1)
else:
    st.warning("⚠️ 'Customer_Age' column not found in the dataset. Skipping age distribution chart.")

# --- 📌 CHART 2: Card Category Distribution ---
st.header("🎴 Card Category Distribution")

if "Card_Category" in c_data.columns and "Gender" in c_data.columns:
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Platinum Card Holders', 'Blue Card Holders'],
        specs=[[{"type": "domain"}, {"type": "domain"}]]
    )

    # Platinum Card Holders by Gender
    fig2.add_trace(
        go.Pie(
            labels=['Female Platinum Card Holders', 'Male Platinum Card Holders'],
            values=c_data.query('Card_Category=="Platinum"').Gender.value_counts().values,
            pull=[0, 0.05],
            hole=0.3
        ),
        row=1, col=1
    )

    # Blue Card Holders by Gender
    fig2.add_trace(
        go.Pie(
            labels=['Female Blue Card Holders', 'Male Blue Card Holders'],
            values=c_data.query('Card_Category=="Blue"').Gender.value_counts().values,
            pull=[0, 0.05],
            hole=0.3
        ),
        row=1, col=2
    )

    fig2.update_layout(height=800, showlegend=True, title_text="<b>Distribution Of Different Card Statuses<b>")
    st.plotly_chart(fig2)
else:
    st.warning("⚠️ 'Card_Category' or 'Gender' column not found. Skipping card distribution chart.")

# --- 📌 CHART 3: General Card Category Distribution ---
if "Card_Category" in c_data.columns:
    fig3 = px.pie(
        c_data,
        height=800,
        names="Card_Category",
        title="Proportion Of Different Card Categories",
        hole=0.3,
    )
    st.plotly_chart(fig3)
else:
    st.warning("⚠️ 'Card_Category' column not found. Skipping this chart.")

# --- 📌 CHART 4: Churn Distribution ---
if "Attrition_Flag" in c_data.columns:
    fig4 = px.pie(
        c_data,
        height=800,
        names='Attrition_Flag',
        title='Proportion of churn vs not churn customers',
        hole=0.3
    )
    st.plotly_chart(fig4)
else:
    st.warning("⚠️ 'Attrition_Flag' column not found. Skipping churn analysis.")

# --- 📌 Data Preprocessing ---
try:
    c_data.Attrition_Flag = c_data.Attrition_Flag.replace({'Attrited Customer': 1, 'Existing Customer': 0})
    c_data.Gender = c_data.Gender.replace({'F': 1, 'M': 0})
    c_data = pd.concat([c_data, pd.get_dummies(c_data['Education_Level']).drop(columns=['Unknown'], errors='ignore')], axis=1)
    c_data = pd.concat([c_data, pd.get_dummies(c_data['Income_Category']).drop(columns=['Unknown'], errors='ignore')], axis=1)
    c_data = pd.concat([c_data, pd.get_dummies(c_data['Marital_Status']).drop(columns=['Unknown'], errors='ignore')], axis=1)
    c_data = pd.concat([c_data, pd.get_dummies(c_data['Card_Category']).drop(columns=['Platinum'], errors='ignore')], axis=1)
    c_data.drop(columns=['Education_Level', 'Income_Category', 'Marital_Status', 'Card_Category', 'CLIENTNUM'], inplace=True, errors='ignore')

    # Oversampling
    oversample = SMOTE()
    X, y = oversample.fit_resample(c_data.iloc[:, 1:], c_data.iloc[:, 0])
    usampled_df = X.assign(Churn=y)

    # PCA
    N_COMPONENTS = 4
    pca_model = PCA(n_components=N_COMPONENTS)
    pc_matrix = pca_model.fit_transform(usampled_df.iloc[:, 15:-1])
    usampled_df_with_pcs = pd.concat([usampled_df, pd.DataFrame(pc_matrix, columns=[f'PC-{i}' for i in range(N_COMPONENTS)])], axis=1)

    # Model Selection
    X_features = ['Total_Trans_Ct', 'PC-3', 'PC-1', 'PC-0', 'PC-2', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count']
    X = usampled_df_with_pcs[X_features]
    y = usampled_df_with_pcs['Churn']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42)

    # Cross-validation
    rf_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", RandomForestClassifier(random_state=42))])
    ada_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", AdaBoostClassifier(random_state=42, learning_rate=0.7))])
    svm_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", SVC(random_state=42, kernel='rbf'))])

    # Model Training and Evaluation
    rf_pipe.fit(train_x, train_y)
    ada_pipe.fit(train_x, train_y)
    svm_pipe.fit(train_x, train_y)

    f1_scores = {
        "Random Forest": f1(test_y, rf_pipe.predict(test_x)),
        "AdaBoost": f1(test_y, ada_pipe.predict(test_x)),
        "SVM": f1(test_y, svm_pipe.predict(test_x))
    }

    df_scores = pd.DataFrame({"Model": list(f1_scores.keys()), "F1 Score": list(f1_scores.values())})
    fig = px.bar(df_scores, x="Model", y="F1 Score", title="F1 Scores for Different Models", color="Model")
    st.plotly_chart(fig)

except Exception as e:
    st.error(f"❌ Error in data processing: {e}")
