# Save this as Home.py in the root directory

import streamlit as st

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

st.title("📊 Welcome to Customer Data Analysis Dashboard")
st.write("Use the sidebar to navigate between pages:")
st.write("- Customer Data Analysis: Explore customer data with visualizations")
st.write("- CSV ChatBot: Interact with your data using AI")