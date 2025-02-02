import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1, recall_score as recall
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
import google.generativeai as genai
from typing import Optional

# Set page config once at the very beginning
st.set_page_config(
    page_title="Customer Data Analysis", 
    layout="wide"
)

# Direct GIF URL (ensure it's accessible)
gif_url = "https://erasebg.org/media/uploads/wp2757874-wallpaper-gif.gif"

# Inject CSS for Full-Screen Background GIF
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

    .overlay {{
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }}
    </style>
"""
st.markdown(bg_style, unsafe_allow_html=True)

# Initialize Gemini API
GEMINI_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

class CSVChatBot:
    def __init__(self):
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-pro')
        self.df: Optional[pd.DataFrame] = None
        self.context: Optional[str] = None
    
    def load_csv(self, file) -> str:
        try:
            self.df = pd.read_csv(file)
            columns_info = "\n".join([
                f"- {col}: {self.df[col].dtype}" 
                for col in self.df.columns
            ])
            numeric_stats = self.df.describe().to_string()
            
            self.context = f"""
            This CSV file has {len(self.df)} rows and {len(self.df.columns)} columns.
            
            Columns information:
            {columns_info}
            
            Basic statistics for numeric columns:
            {numeric_stats}
            
            First few rows of data:
            {self.df.head().to_string()}
            """
            return "CSV file loaded successfully!"
        except Exception as e:
            return f"Error loading CSV: {str(e)}"
    
    def get_response(self, user_query: str) -> str:
        if self.df is None:
            return "Please load a CSV file first!"
        
        try:
            prompt = f"""
            Context about the CSV data:
            {self.context}
            
            User Question: {user_query}
            
            Please provide a clear and concise answer based on the CSV data above.
            If the question can be answered with a visualization, suggest what type of chart would be most appropriate.
            If calculation is involved provide the formula used along with the output
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error getting response: {str(e)}"

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'df' not in st.session_state:
        st.session_state.df = None

@st.cache_data
def load_data():
    data = pd.read_csv("BankChurners.csv")
    data = data[data.columns[:-2]]
    return data

def main():
    st.title("📊 Customer Data Analysis Dashboard")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize chatbot if not already done
    if st.session_state.chatbot is None:
        st.session_state.chatbot = CSVChatBot()

    # Load initial data
    c_data = load_data()

    # Show dataset preview if checked
    if st.checkbox("Show raw data"):
        st.write(c_data.head())

    # Add your visualizations here
    
    # Sidebar for CSV upload and chat
    with st.sidebar:
        st.header("Upload Data for AI Analysis")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            if st.button("Load CSV"):
                result = st.session_state.chatbot.load_csv(uploaded_file)
                st.success(result)
                st.session_state.df = st.session_state.chatbot.df
        
        if st.session_state.df is not None:
            st.header("Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # Chat interface
    if st.session_state.df is not None:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("Ask about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()