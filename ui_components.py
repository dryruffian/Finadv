import streamlit as st
import base64

def render_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700&family=Inter:wght@300;400;600&display=swap');

        :root {
            --primary-color: #00F5FF;
            --secondary-color: #FF00E4;
            --bg-color: #0A0E17;
            --text-color: #E0E0E0;
            --card-bg: #141C2F;
        }

        body {
            color: var(--text-color);
            background-color: var(--bg-color);
            font-family: 'Inter', sans-serif;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 245, 255, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(255, 0, 228, 0.1) 0%, transparent 20%);
            background-attachment: fixed;
        }

        .stApp {
            background: transparent;
        }

        h1, h2, h3 {
            font-family: 'Exo 2', sans-serif;
            color: var(--primary-color);
            text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
            letter-spacing: 1px;
        }

        .stButton > button {
            font-family: 'Exo 2', sans-serif;
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: var(--bg-color);
            font-weight: 700;
            border-radius: 30px;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 245, 255, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 8px 20px rgba(255, 0, 228, 0.6);
        }

        /* Add more custom CSS as needed */

    </style>
    """, unsafe_allow_html=True)

def render_header():
    # Load and encode the logo image
    with open("New_logo.png", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    st.markdown(
        f'''
        <style>
        .header {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 20px;
        }}
        .header img {{
            max-width: 100%;
            height: auto;
            width: 80px; /* Default width */
        }}
        @media (min-width: 600px) {{
            .header h1 {{
                font-size: 2.5em;
            }}
            .header img {{
                width: 100px; /* Image size for wider screens */
            }}
        }}
        @media (max-width: 599px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .header img {{
                width: 80px; /* Image size for smaller screens */
            }}
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{image_base64}" />
            <h1>MarketWealth Genius: Your AI Financial Advisor</h1>
        </div>
        ''',
        unsafe_allow_html=True
    )