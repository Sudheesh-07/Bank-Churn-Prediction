import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
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

# # Load the dataset
@st.cache_data
def load_data():
    # file_path = r"C:\Users\nadar\Downloads\BankChurners.csv"  # Ensure correct file path
    data = pd.read_csv("BankChurners.csv")
    data = data[data.columns[:-2]]  # Drop last two columns
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

#Data Preprocessing part
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

N_COMPONENTS = 4

pca_model = PCA(n_components = N_COMPONENTS )

pc_matrix = pca_model.fit_transform(ohe_data)

evr = pca_model.explained_variance_ratio_
total_var = evr.sum() * 100
cumsum_evr = np.cumsum(evr)

trace1 = {
    "name": "individual explained variance", 
    "type": "bar", 
    'y':evr}
trace2 = {
    "name": "cumulative explained variance", 
    "type": "scatter", 
     'y':cumsum_evr}
data = [trace1, trace2]
layout = {
    "xaxis": {"title": "Principal components"}, 
    "yaxis": {"title": "Explained variance ratio"},
  }

usampled_df_with_pcs = pd.concat([usampled_df,pd.DataFrame(pc_matrix,columns=['PC-{}'.format(i) for i in range(0,N_COMPONENTS)])],axis=1)


# Pearson Correlation and Spearman Correlation
s_val =usampled_df_with_pcs.corr('pearson')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values

s_val =usampled_df_with_pcs.corr('spearman')
s_idx = s_val.index
s_col = s_val.columns
s_val = s_val.values


#Model Selection and evaluation
X_features = ['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2','Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']

X = usampled_df_with_pcs[X_features]
y = usampled_df_with_pcs['Churn']

train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=42)

#Cross Vaidation

rf_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",RandomForestClassifier(random_state=42)) ])
ada_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",AdaBoostClassifier(random_state=42,learning_rate=0.7)) ])
svm_pipe = Pipeline(steps =[ ('scale',StandardScaler()), ("RF",SVC(random_state=42,kernel='rbf')) ])


f1_cross_val_scores = cross_val_score(rf_pipe,train_x,train_y,cv=5,scoring='f1')
ada_f1_cross_val_scores=cross_val_score(ada_pipe,train_x,train_y,cv=5,scoring='f1')
svm_f1_cross_val_scores=cross_val_score(svm_pipe,train_x,train_y,cv=5,scoring='f1')


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
fig = px.bar(
    df_scores.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
    x="Model", 
    y="Score", 
    color="Metric", 
    barmode="group", 
    title="F1 Score and Recall Score for Different Models")

# Display the plot in Streamlit
st.plotly_chart(fig)

