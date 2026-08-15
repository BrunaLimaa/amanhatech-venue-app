import streamlit as st
import pandas as pd
from core.data_loader import load_training_data
from core.ml_engine import train_models

# 1. Page Configuration
st.set_page_config(page_title="Venue Booking Intelligence", layout="wide")

# 2. Sidebar Configuration & API Input
st.sidebar.header("Configuration")
jambase_key_input = st.sidebar.text_input(
    "JamBase API Key (Optional)", 
    type="password", 
    value=""
)

# 3. Model Pipeline Initialization (Cached for performance)
@st.cache_resource(show_spinner="Loading data and training models...")
def initialize_pipeline(api_key):
    api_key_clean = api_key.strip() if api_key else None
    X, y_price, y_sellout = load_training_data(jambase_api_key=api_key_clean)
    regressor, classifier = train_models(X, y_price, y_sellout)
    return regressor, classifier

regressor, classifier = initialize_pipeline(jambase_key_input)

# 4. Sidebar Booking Parameters
st.sidebar.header("Booking Parameters")
st.sidebar.markdown("Configure venue capacity, audience demand, and local economic indexes.")

input_capacity = st.sidebar.slider("Target Venue Capacity", min_value=300, max_value=5000, value=1500, step=100)
input_demand = st.sidebar.slider("Artist Audience Demand (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
input_ibge = st.sidebar.slider("IBGE Economic Index (0.5 - 1.5)", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

# 5. Model Inference & Financial Simulation
input_data = pd.DataFrame({
    'capacity': [input_capacity],
    'spotify_demand': [input_demand],
    'ibge_index': [input_ibge]
})

predicted_price = regressor.predict(input_data)[0]
sellout_probability = classifier.predict_proba(input_data)[0][1]
projected_revenue = predicted_price * input_capacity * sellout_probability

# 6. Main Dashboard Interface
st.title("Venue Booking Intelligence & Revenue Simulation")
st.markdown("A B2B platform leveraging multi-source data to optimize tour routing and maximize box office margins for independent venues.")

st.subheader("Financial Simulation Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Optimal Ticket Price", value=f"${predicted_price:.2f}")

with col2:
    st.metric(label="Sell-Out Probability", value=f"{sellout_probability * 100:.1f}%")

with col3:
    st.metric(label="Projected Gross Revenue", value=f"${projected_revenue:,.2f}")

st.divider()

st.markdown("### System Architecture")
st.markdown("* **Pricing Optimization:** Random Forest Regressor trained on cross-referenced audience density and IBGE purchasing power.")
st.markdown("* **Risk & Sell-Out Classification:** Random Forest Classifier assessing venue capacity constraints against local market density.")
st.markdown("* **Data Integration:** Combines Last.fm listening profiles, JamBase touring histories, and public IBGE socioeconomic data.")