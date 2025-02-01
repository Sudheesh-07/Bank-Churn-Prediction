import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
import io
import json

app = Flask(__name__)

def process_data(data):
    # Create a copy to avoid modifying the original dataframe
    data = data.copy()
    
    # Data preprocessing
    data = data[data.columns[:-2]]  # Drop last two columns
    data.Attrition_Flag = data.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
    data.Gender = data.Gender.replace({'F':1,'M':0})
    
    # One-hot encoding with error handling
    categorical_columns = ['Education_Level', 'Income_Category', 'Marital_Status', 'Card_Category']
    drop_values = ['Unknown', 'Unknown', 'Unknown', 'Platinum']
    
    for col, drop_val in zip(categorical_columns, drop_values):
        if col in data.columns:
            dummies = pd.get_dummies(data[col])
            if drop_val in dummies.columns:
                dummies = dummies.drop(columns=[drop_val])
            data = pd.concat([data, dummies], axis=1)
    
    # Drop original categorical columns and CLIENTNUM
    columns_to_drop = categorical_columns + ['CLIENTNUM']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    return data

def generate_charts(data):
    # Age Distribution
    fig1 = make_subplots(rows=2, cols=1)
    tr1 = go.Box(x=data['Customer_Age'].tolist(), name='Age Box Plot', boxmean=True)
    tr2 = go.Histogram(x=data['Customer_Age'].tolist(), name='Age Histogram')
    fig1.add_trace(tr1, row=1, col=1)
    fig1.add_trace(tr2, row=2, col=1)
    fig1.update_layout(height=700, width=1200, title_text="Distribution of Customer Ages")
    
    # Card Category Distribution by Gender
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Platinum Card Holders', 'Blue Card Holders'],
        specs=[[{"type": "domain"}, {"type": "domain"}]]
    )
    
    # Platinum Card Holders by Gender
    platinum_data = data[data['Card_Category']=="Platinum"]['Gender'].value_counts()
    fig2.add_trace(
        go.Pie(
            labels=['Female Platinum Card Holders', 'Male Platinum Card Holders'],
            values=platinum_data.values.tolist(),
            hole=0.3
        ),
        row=1, col=1
    )
    
    # Blue Card Holders by Gender
    blue_data = data[data['Card_Category']=="Blue"]['Gender'].value_counts()
    fig2.add_trace(
        go.Pie(
            labels=['Female Blue Card Holders', 'Male Blue Card Holders'],
            values=blue_data.values.tolist(),
            hole=0.3
        ),
        row=1, col=2
    )
    
    # Card Category Distribution
    fig3 = px.pie(data, names="Card_Category", title="Proportion Of Different Card Categories", hole=0.3)
    
    # Churn Distribution
    fig4 = px.pie(data, names='Attrition_Flag', title='Proportion of churn vs not churn customers', hole=0.3)
    
    # Convert to JSON-serializable format
    charts = {
        'age_dist': json.loads(fig1.to_json()),
        'gender_dist': json.loads(fig2.to_json()),
        'card_dist': json.loads(fig3.to_json()),
        'churn_dist': json.loads(fig4.to_json())
    }
    
    return charts

def train_models(data):
    try:
        oversample = SMOTE()
        X = data[data.columns[1:]].copy()
        y = data[data.columns[0]].copy()
        X, y = oversample.fit_resample(X, y)
        usampled_df = pd.DataFrame(X, columns=data.columns[1:])
        usampled_df['Churn'] = y
        
        ohe_data = usampled_df[usampled_df.columns[15:-1]].copy()
        usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])
        
        # PCA
        N_COMPONENTS = 4
        pca_model = PCA(n_components=N_COMPONENTS)
        pc_matrix = pca_model.fit_transform(ohe_data)
        pc_cols = [f'PC-{i}' for i in range(N_COMPONENTS)]
        usampled_df_with_pcs = pd.concat([
            usampled_df,
            pd.DataFrame(pc_matrix, columns=pc_cols)
        ], axis=1)
        
        # Model training
        X_features = ['Total_Trans_Ct','PC-3','PC-1','PC-0','PC-2',
                     'Total_Ct_Chng_Q4_Q1','Total_Relationship_Count']
        X = usampled_df_with_pcs[X_features]
        y = usampled_df_with_pcs['Churn']
        
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=42)
        
        models = {
            'Random Forest': Pipeline([
                ('scale', StandardScaler()),
                ("RF", RandomForestClassifier(random_state=42))
            ]),
            'AdaBoost': Pipeline([
                ('scale', StandardScaler()),
                ("ADA", AdaBoostClassifier(random_state=42, learning_rate=0.7))
            ]),
            'SVM': Pipeline([
                ('scale', StandardScaler()),
                ("SVM", SVC(random_state=42, kernel='rbf'))
            ])
        }
        
        results = {}
        for name, model in models.items():
            model.fit(train_x, train_y)
            predictions = model.predict(test_x)
            results[name] = float(f1(test_y, predictions))
        
        return results
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        data = pd.read_csv(stream)
        
        # Generate initial charts from raw data
        charts = generate_charts(data)
        
        # Process data and train models
        processed_data = process_data(data)
        model_results = train_models(processed_data)
        
        return jsonify({
            'charts': charts,
            'model_results': model_results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)