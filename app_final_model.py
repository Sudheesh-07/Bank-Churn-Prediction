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
from google import genai
from typing import Optional

# Streamlit Page Config
# Set up Streamlit page configuration
st.set_page_config(page_title="Customer Data Analysis", layout="wide")

# Direct GIF URL (ensure it's accessible)
gif_url = "https://erasebg.org/media/uploads/wp2757874-wallpaper-gif.gif"

# Inject CSS for Full-Screen Background GIF, covering the entire page including borders
bg_style = f"""
    <style>
    .stApp {{
        background: url('{gif_url}') no-repeat center center fixed;
        background-size: cover;
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    /* Optional: Text Readability Overlay */
    .overlay {{
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }}
    </style>
"""
st.markdown(bg_style, unsafe_allow_html=True)

# Title of the application with an overlay
st.markdown('<div class="overlay"><h1>Customer Data Analysis Dashboard</h1></div>', unsafe_allow_html=True)


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

def add_sentiment_features(df):
    """
    Add sentiment-based categorical features to the dataset with error handling for missing columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with original features
    
    Returns:
    pd.DataFrame: DataFrame with new sentiment features
    """
    required_columns = [
        'Months_Inactive_12_mon', 'Total_Relationship_Count', 'Total_Trans_Ct',
        'Total_Revolving_Bal', 'Credit_Limit', 'Total_Amt_Chng_Q4_Q1',
        'Total_Ct_Chng_Q4_Q1', 'Contacts_Count_12_mon', 'Total_Trans_Amt',
        'Months_on_book'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Error: The following required columns are missing from your data: {', '.join(missing_columns)}")
        st.write("Please make sure your CSV file contains all required columns:")
        st.write(required_columns)
        return None
    
    df = df.copy()
    
    # 1. Customer Engagement Level
    def calculate_engagement_score(row):
        try:
            engagement_score = (
                (row['Total_Relationship_Count'] / 6) * 0.3 +
                (row['Total_Trans_Ct'] / df['Total_Trans_Ct'].max()) * 0.4 +
                (1 - row['Months_Inactive_12_mon'] / 12) * 0.3
            )
            if engagement_score >= 0.7:
                return "Highly_Engaged"
            elif engagement_score >= 0.4:
                return "Moderately_Engaged"
            elif engagement_score >= 0.2:
                return "Passively_Engaged"
            else:
                return "Disengaged"
        except Exception as e:
            st.warning(f"Warning: Error calculating engagement score: {str(e)}")
            return "Unknown"
    
    df['Engagement_Level'] = df.apply(calculate_engagement_score, axis=1)
    
    # 2. Product Usage Pattern
    def determine_usage_pattern(row):
        try:
            if row['Total_Trans_Ct'] > df['Total_Trans_Ct'].quantile(0.75):
                return "Power_User"
            elif row['Total_Trans_Ct'] > df['Total_Trans_Ct'].quantile(0.5):
                return "Regular_User"
            elif row['Total_Trans_Ct'] > df['Total_Trans_Ct'].quantile(0.25):
                return "Occasional_User"
            else:
                return "Minimal_User"
        except Exception as e:
            st.warning(f"Warning: Error determining usage pattern: {str(e)}")
            return "Unknown"
    
    df['Usage_Pattern'] = df.apply(determine_usage_pattern, axis=1)
    
    # 3. Financial Behavior
    def analyze_financial_behavior(row):
        try:
            utilization = row['Total_Revolving_Bal'] / row['Credit_Limit']
            spending_change = row['Total_Amt_Chng_Q4_Q1']
            
            if utilization < 0.3 and spending_change > 0.5:
                return "Financially_Savvy"
            elif utilization < 0.5 and spending_change > 0:
                return "Balanced_Spender"
            elif utilization >= 0.5 and utilization < 0.8:
                return "Credit_Dependent"
            else:
                return "High_Risk"
        except Exception as e:
            st.warning(f"Warning: Error analyzing financial behavior: {str(e)}")
            return "Unknown"
    
    df['Financial_Behavior'] = df.apply(analyze_financial_behavior, axis=1)
    
    # 4. Customer Value Segment
    def segment_customer_value(row):
        try:
            value_score = (
                (row['Total_Trans_Amt'] / df['Total_Trans_Amt'].max()) * 0.4 +
                (row['Credit_Limit'] / df['Credit_Limit'].max()) * 0.3 +
                (row['Total_Relationship_Count'] / 6) * 0.3
            )
            if value_score >= 0.7:
                return "Premium"
            elif value_score >= 0.4:
                return "Mid_Tier"
            else:
                return "Basic"
        except Exception as e:
            st.warning(f"Warning: Error segmenting customer value: {str(e)}")
            return "Unknown"
    
    df['Value_Segment'] = df.apply(segment_customer_value, axis=1)
    
    # 5. Risk Level
    def assess_risk_level(row):
        try:
            risk_factors = 0
            if row['Months_Inactive_12_mon'] > 3:
                risk_factors += 1
            if row['Total_Ct_Chng_Q4_Q1'] < 0:
                risk_factors += 1
            if row['Total_Revolving_Bal'] / row['Credit_Limit'] > 0.7:
                risk_factors += 1
            if row['Contacts_Count_12_mon'] > 3:
                risk_factors += 1
                
            if risk_factors >= 3:
                return "High_Risk"
            elif risk_factors == 2:
                return "Medium_Risk"
            elif risk_factors == 1:
                return "Low_Risk"
            else:
                return "Minimal_Risk"
        except Exception as e:
            st.warning(f"Warning: Error assessing risk level: {str(e)}")
            return "Unknown"
    
    df['Risk_Level'] = df.apply(assess_risk_level, axis=1)
    
    return df

c_data.Attrition_Flag = c_data.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
c_data.Gender = c_data.Gender.replace({'F':1,'M':0})
c_data = pd.concat([c_data,pd.get_dummies(c_data['Education_Level']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Income_Category']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Marital_Status']).drop(columns=['Unknown'])],axis=1)
c_data = pd.concat([c_data,pd.get_dummies(c_data['Card_Category']).drop(columns=['Platinum'])],axis=1)
c_data.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category','CLIENTNUM'],inplace=True)
#c_data = add_sentiment_features(c_data)
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
ada_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",AdaBoostClassifier(random_state=42,learning_rate=0.7)) ])
svm_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",SVC(random_state=42,kernel='rbf')) ])

# Save the trained model
joblib.dump(rf_pipe, 'churn_prediction_model.pkl')

# Load the model
model = joblib.load('churn_prediction_model.pkl')


#Model Evaluation
rf_pipe.fit(train_x,train_y)
rf_prediction = rf_pipe.predict(test_x)

ada_pipe.fit(train_x,train_y)
ada_prediction = ada_pipe.predict(test_x)

svm_pipe.fit(train_x,train_y)
svm_prediction = svm_pipe.predict(test_x)
f1_scores = {
    "Random Forest": f1(test_y, rf_prediction),
    "AdaBoost": f1(test_y, ada_prediction),
    "SVM": f1(test_y, svm_prediction)
}

recall_scores = {
    "Random Forest": recall(test_y, rf_prediction),
    "AdaBoost": recall(test_y, ada_prediction),
    "SVM": recall(test_y, svm_prediction)
}

# Creating a DataFrame for visualization
df_scores = pd.DataFrame({"Model": list(f1_scores.keys()), "F1 Score": list(f1_scores.values()), "Recall Score": list(recall_scores.values())})

# Creating a bar plot
fig5 = px.bar(
    df_scores.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
    x="Model", 
    y="Score", 
    color="Metric", 
    barmode="group", 
    title="F1 Score and Recall Score for Different Models",
    text="Score"
    )
fig5.update_traces(texttemplate='%{y:.2f}', textposition='inside')
# Display the plot in Streamlit
st.plotly_chart(fig5)
# Upload CSV and predict churn
st.sidebar.header("Upload your CSV file")
file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])


if file:
    user_data = pd.read_csv(file)

    # Add sentiment-based features
    user_data = add_sentiment_features(user_data)

    # Preprocess user data for prediction
    user_data_processed = user_data[X_features]
    predictions = model.predict(user_data_processed)
    user_data['Churn_Prediction'] = predictions

    # Filter churned users
    churned_users = user_data[user_data['Churn_Prediction'] == 1]

    # Churn Sentiment Analysis
    st.header("📊 Sentiment Analysis of Churned Users")

    if churned_users.empty:
        st.write("✅ No churned users found in the uploaded data.")
    else:
        # Distribution of sentiment-based features among churned customers
        sentiment_features = ['Engagement_Level', 'Usage_Pattern', 'Financial_Behavior', 'Value_Segment', 'Risk_Level']
        for feature in sentiment_features:
            fig = px.histogram(churned_users, x=feature, title=f"Distribution of {feature} Among Churned Customers")
            st.plotly_chart(fig)

        # Analysis on why users are churning
        churn_reasons = churned_users['Risk_Level'].value_counts().reset_index()
        churn_reasons.columns = ['Risk Level', 'Count']
        st.write("### 🔍 Major Risk Factors for Churn")
        st.write(churn_reasons)

        # Insights into churn reasons
        def analyze_churn_reason(row):
            if row['Risk_Level'] in ['High_Risk', 'Medium_Risk']:
                return "🔴 High Risk: Likely due to financial instability or lack of engagement."
            elif row['Engagement_Level'] in ['Disengaged', 'Passively_Engaged']:
                return "🟠 Low Engagement: Customers are not interacting enough with the bank."
            elif row['Usage_Pattern'] in ['Minimal_User', 'Occasional_User']:
                return "🟡 Low Usage: Customers are not actively using banking services."
            else:
                return "🟢 Other Factors"

        churned_users['Churn_Reason'] = churned_users.apply(analyze_churn_reason, axis=1)

        # Display churned users with analysis
        st.write("### 📋 Churned Customers with Predicted Reasons")
        st.write(churned_users[['Churn_Prediction', 'Engagement_Level', 'Usage_Pattern', 'Financial_Behavior', 'Value_Segment', 'Risk_Level', 'Churn_Reason']])
    

#