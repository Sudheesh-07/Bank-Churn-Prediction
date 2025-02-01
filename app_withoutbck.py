import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1, recall_score as recall
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
# Streamlit Page Config
st.set_page_config(layout="wide", page_title="Bank Churn Analysis")

@st.cache_data
def load_data():
    data = pd.read_csv("BankChurners.csv")
    data = data[data.columns[:-2]]
    return data


# Load Data
c_data = load_data()

# Title
st.title("📊 Bank Customer Churn Analysis")

# Show dataset preview if checked
if st.checkbox("Show raw data"):
    st.write(c_data.head())

# --- 📌 CHART 1: Age Distribution ---
st.header("📈 Distribution of Customer Ages")

# Create subplot for age distribution
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

st.header(" Card Category Distribution")

# Create subplot for gender distribution
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Platinum Card Holders', 'Blue Card Holders'],  # Fixed number of titles
    specs=[
        [{"type": "domain"}, {"type": "domain"}]  # Corrected layout
    ]
)

# Platinum Card Holders by Gender
fig2.add_trace(
    go.Pie(
        labels=['Female Platinum Card Holders', 'Male Platinum Card Holders'],
        values=c_data.query('Card_Category=="Platinum"').Gender.value_counts().values,
        pull=[0, 0.05],  # Matching length with labels
        hole=0.3
    ),
    row=1, col=1
)

# Blue Card Holders by Gender
fig2.add_trace(
    go.Pie(
        labels=['Female Blue Card Holders', 'Male Blue Card Holders'],
        values=c_data.query('Card_Category=="Blue"').Gender.value_counts().values,
        pull=[0, 0.05],  # Matching length with labels
        hole=0.3
    ),
    row=1, col=2
)

# Update layout
fig2.update_layout(
    height=800,
    showlegend=True,
    title_text="<b>Distribution Of Different Card Statuses<b>",
)

# Display plot in Streamlit
st.plotly_chart(fig2)

fig3 = px.pie(
    c_data,
    height=800,
    names="Card_Category",
    title="Proportion Of Different Card Categories",
    hole=0.3,
)

st.plotly_chart(fig3)

fig4 = px.pie(
    c_data,
    height=800,
    names='Attrition_Flag',
    title='Proportion of churn vs not churn customers',
    hole=0.3
)

st.plotly_chart(fig4)

c_data = load_data()
c_data.Attrition_Flag = c_data.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
c_data.Gender = c_data.Gender.replace({'F':1,'M':0})
c_data = pd.concat([c_data,pd.get_dummies(c_data['Education_Level']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Income_Category']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Marital_Status']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Card_Category']).drop(columns=['Platinum'])],axis=1)
c_data.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category','CLIENTNUM'],inplace=True)

oversample = SMOTE()
X, y = oversample.fit_resample(c_data[c_data.columns[1:]], c_data[c_data.columns[0]])
usampled_df = X.assign(Churn = y)
ohe_data =usampled_df[usampled_df.columns[15:-1]].copy()
usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])

# New Features
usampled_df['Credit_Utilization'] = usampled_df['Total_Revolving_Bal'] / usampled_df['Credit_Limit']
usampled_df['CLV'] = usampled_df['Total_Trans_Amt'] * usampled_df['Months_on_book']
usampled_df['Inactive_Months_Ratio'] = usampled_df['Months_Inactive_12_mon'] / usampled_df['Months_on_book']

N_COMPONENTS = 4
pca_model = PCA(n_components = N_COMPONENTS )
pc_matrix = pca_model.fit_transform(ohe_data)
usampled_df_with_pcs = pd.concat([usampled_df,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)

X_features = ['Total_Trans_Ct', 'PC-3', 'PC-1', 'PC-0', 'PC-2', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count',
              'Credit_Utilization', 'CLV', 'Inactive_Months_Ratio']
X = usampled_df_with_pcs[X_features]
y = usampled_df_with_pcs['Churn']
train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)

rf_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",RandomForestClassifier(random_state=42)) ])
rf_pipe.fit(train_x,train_y)

# Save the trained model
joblib.dump(rf_pipe, 'churn_prediction_model.pkl')

# Load the model
model = joblib.load('churn_prediction_model.pkl')

# Upload CSV and predict churn
st.header("📂 Upload Customer Data for Churn Prediction")
file = st.file_uploader("Upload CSV file", type=["csv"])
if file:
    user_data = pd.read_csv(file)
    user_data_processed = user_data[X_features]
    predictions = model.predict(user_data_processed)
    user_data['Churn_Prediction'] = predictions
    
    def categorize_customer(row):
        if row['Churn_Prediction'] == 1:
            if row['Total_Trans_Ct'] < 40 or row['Total_Ct_Chng_Q4_Q1'] < 0.5:
                return "🅰 High Priority (Immediate Action)"
            elif row['Credit_Limit'] < 5000 or row['Total_Revolving_Bal'] > 2000:
                return "🅱 Medium Priority (Financial Risk)"
            else:
                return "🅲 Low Priority (Long-term Monitoring)"
        else:
            return "No Churn Risk"
    
    user_data['Churn_Category'] = user_data.apply(categorize_customer, axis=1)
    st.write(user_data)
    
    category_counts = user_data['Churn_Category'].value_counts()
    fig = px.pie(category_counts, names=category_counts.index, values=category_counts.values, title="Churn Risk Category Distribution",hole=0.3)
    st.plotly_chart(fig)
