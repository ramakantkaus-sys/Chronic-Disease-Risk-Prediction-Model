import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="Depression & Chronic Disease Risk Assessment",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("decision_treedepression.pkl", "rb"))

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Mappings
MAPPINGS = {
  "Marital Status": {
    "Divorced": 0, "Married": 1, "Single": 2, "Widowed": 3
  },
  "Education Level": {
    "Associate Degree": 0, "Bachelor's Degree": 1, "High School": 2, "Master's Degree": 3, "PhD": 4
  },
  "Smoking Status": {
    "Current": 0, "Former": 1, "Non-smoker": 2
  },
  "Physical Activity Level": {
    "Active": 0, "Moderate": 1, "Sedentary": 2
  },
  "Employment Status": {
    "Employed": 0, "Unemployed": 1
  },
  "Alcohol Consumption": {
    "High": 0, "Low": 1, "Moderate": 2
  },
  "Dietary Habits": {
    "Healthy": 0, "Moderate": 1, "Unhealthy": 2
  },
  "Sleep Patterns": {
    "Fair": 0, "Good": 1, "Poor": 2
  },
  "History of Mental Illness": {
    "No": 0, "Yes": 1
  },
  "History of Substance Abuse": {
    "No": 0, "Yes": 1
  },
  "Family History of Depression": {
    "No": 0, "Yes": 1
  }
}

def preprocess_input(data):
    """Convert inputs into numerical values using the mappings."""
    vector = []
    
    # Numerical Fields directly
    vector.append(data["Age"])
    
    # Categorical Fields with Mapping
    vector.append(MAPPINGS["Marital Status"][data["Marital Status"]])
    vector.append(MAPPINGS["Education Level"][data["Education Level"]])
    
    # Numerical
    vector.append(data["Number of Children"])
    
    # Categorical
    vector.append(MAPPINGS["Smoking Status"][data["Smoking Status"]])
    vector.append(MAPPINGS["Physical Activity Level"][data["Physical Activity Level"]])
    vector.append(MAPPINGS["Employment Status"][data["Employment Status"]])
    
    # Numerical
    vector.append(data["Income"])
    
    # Categorical
    vector.append(MAPPINGS["Alcohol Consumption"][data["Alcohol Consumption"]])
    vector.append(MAPPINGS["Dietary Habits"][data["Dietary Habits"]])
    vector.append(MAPPINGS["Sleep Patterns"][data["Sleep Patterns"]])
    vector.append(MAPPINGS["History of Mental Illness"][data["History of Mental Illness"]])
    vector.append(MAPPINGS["History of Substance Abuse"][data["History of Substance Abuse"]])
    vector.append(MAPPINGS["Family History of Depression"][data["Family History of Depression"]])
    
    return np.array(vector).reshape(1, -1)

import base64
import random

# ... (Previous imports)

# Function to encode image to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Select random background
background_files = ["1.avif", "2.jpeg", "3.jpeg"]
selected_bg = random.choice(background_files)

try:
    bg_img = get_base64(selected_bg)
    bg_ext = selected_bg.split('.')[-1]
except Exception as e:
    # Fallback if image not found
    bg_img = ""
    bg_ext = "jpeg"
    st.error(f"Error loading background: {e}")

# UI Implementation using Custom CSS
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-image: url("data:image/{bg_ext};base64,{bg_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Overlay for readability */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.4); 
        z-index: -1;
    }}

    /* Container Styling */
    .css-1r6slb0, .stForm {{
        background-color: rgba(255, 255, 255, 0.85); /* Slightly more opaque */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }}

    /* Title Styling */
    h1 {{
        color: skyblue;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Increased contrast for visibility */
    }}
    
    /* Subheader/Text Styling */
    .stMarkdown p {{
        font-size: 1.1em;
        color: #2c3e50;
        font-weight: 500;
    }}

    /* Input Fields Styling */
    .stNumberInput, .stSelectbox {{
        margin-bottom: 15px;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white;
        width: 100%;
        padding: 12px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 25px;
        transition: all 0.3s ease 0s;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    }}

    .stButton > button:hover {{
        background: linear-gradient(to right, #2575fc 0%, #6a11cb 100%);
        box-shadow: 0px 15px 20px rgba(46, 229, 157, 0.4);
        transform: translateY(-2px);
    }}
    
    /* Success/Error Message Styling */
    .stAlert {{
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Chronic Disease Risk Prediction Model")
st.markdown("<p style='text-align: center; color: #555; text-shadow: 1px 1px 2px rgba(255,255,255,0.8);'>Assess lifestyle and health factors to predict risks associated with depression and chronic conditions.</p>", unsafe_allow_html=True)

st.write("---")

with st.expander("ℹ️ **About the Project**"):
    st.markdown("""
    ### 🎯 Goal
    This project utilizes a Decision Tree model to analyze lifestyle, demographic, and medical history data to assess the risk of chronic medical conditions, including depression.
    
    ### 🤖 Model Details
    - **Algorithm:** Decision Tree Classifier
    - **Context:** Built to understand correlations between mental health history, lifestyle (sleep, diet, substance use), and chronic disease outcomes.
    
    ### ⚠️ Disclaimer
    *This tool is for educational purposes only and does not constitute medical advice.*
    """)

st.write("")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        marital_status = st.selectbox("Marital Status", options=list(MAPPINGS["Marital Status"].keys()))
        education_level = st.selectbox("Education Level", options=list(MAPPINGS["Education Level"].keys()))
        children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
        smoking_status = st.selectbox("Smoking Status", options=list(MAPPINGS["Smoking Status"].keys()))
        activity_level = st.selectbox("Physical Activity Level", options=list(MAPPINGS["Physical Activity Level"].keys()))
        employment_status = st.selectbox("Employment Status", options=list(MAPPINGS["Employment Status"].keys()))

    with col2:
        income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
        alcohol_consumption = st.selectbox("Alcohol Consumption", options=list(MAPPINGS["Alcohol Consumption"].keys()))
        dietary_habits = st.selectbox("Dietary Habits", options=list(MAPPINGS["Dietary Habits"].keys()))
        sleep_patterns = st.selectbox("Sleep Patterns", options=list(MAPPINGS["Sleep Patterns"].keys()))
        mental_illness = st.selectbox("History of Mental Illness", options=list(MAPPINGS["History of Mental Illness"].keys()))
        substance_abuse = st.selectbox("History of Substance Abuse", options=list(MAPPINGS["History of Substance Abuse"].keys()))
        family_history = st.selectbox("Family History of Depression", options=list(MAPPINGS["Family History of Depression"].keys()))

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    user_data = {
        "Age": age,
        "Marital Status": marital_status,
        "Education Level": education_level,
        "Number of Children": children,
        "Smoking Status": smoking_status,
        "Physical Activity Level": activity_level,
        "Employment Status": employment_status,
        "Income": income,
        "Alcohol Consumption": alcohol_consumption,
        "Dietary Habits": dietary_habits,
        "Sleep Patterns": sleep_patterns,
        "History of Mental Illness": mental_illness,
        "History of Substance Abuse": substance_abuse,
        "Family History of Depression": family_history
    }
    
    try:
        processed_data = preprocess_input(user_data)
        prediction = model.predict(processed_data)[0]
        
        if prediction == 1:
            st.error("⚠️ **Prediction:** High Risk. This profile is associated with a higher likelihood of chronic disease/depression.")
        else:
            st.success("✅ **Prediction:** Low Risk. This profile is not strongly associated with chronic disease/depression.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
