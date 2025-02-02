# Save this as pages/2_CSV_ChatBot.py

import streamlit as st
import pandas as pd
from google import genai
from typing import Optional

# Direct API key (not recommended for production)
GEMINI_API_KEY = "AIzaSyBE3aaQanFxtxgcFJSe8WPPXWqPQmV5Q8w"

class CSVChatBot:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.df: Optional[pd.DataFrame] = None
        self.context: Optional[str] = None
    
    # Rest of the CSVChatBot class implementation remains the same
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
            
            response = self.client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            
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

st.title("CSV ChatBot")

# Initialize session state
initialize_session_state()

# Initialize chatbot if not already done
if st.session_state.chatbot is None:
    st.session_state.chatbot = CSVChatBot()

# Sidebar for file upload and data preview
with st.sidebar:
    st.header("Upload Data")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        if st.button("Load CSV"):
            result = st.session_state.chatbot.load_csv(uploaded_file)
            st.success(result)
            st.session_state.df = st.session_state.chatbot.df
    
    # Show data info if loaded
    if st.session_state.df is not None:
        st.header("Data Preview")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        st.header("Data Info")
        st.text(f"Rows: {len(st.session_state.df)}")
        st.text(f"Columns: {len(st.session_state.df.columns)}")
        
        # Add basic data analysis
        if st.checkbox("Show Column Statistics"):
            st.write(st.session_state.df.describe())

# Main chat interface
if st.session_state.df is None:
    st.info("Please upload a CSV file in the sidebar to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_response(prompt)
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()