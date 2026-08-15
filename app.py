"""
Amanhã Tech - Venue Booking Intelligence Dashboard
Alpha v2.0 (Historical Dataset Architecture)
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from typing import Tuple, Dict, List

# Import core ML and data modules
from core.data_loader import load_training_data
from core.ml_engine import train_models

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Venue Booking Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Pipeline Initialization (Cache Aware) ---
@st.cache_resource(show_spinner="Booting Data...")
def initialize_pipeline() -> Tuple[any, any, Dict[str, float], List[str]]:
    """
    Initializes the data loader and trains the ML models exclusively on the CSV dataset.
    """
    # The API key parameter is passed as None since we transitioned to historical CSV data
    X, y_price, y_sellout, demand_dict, available_cities = load_training_data(jambase_api_key=None)
    
    # Train Logistic Regression and Ridge Regression pipelines
    regressor, classifier = train_models(X, y_price, y_sellout)
    
    return regressor, classifier, demand_dict, available_cities

# Unpack the pipeline globally for the session
regressor, classifier, real_demand_dict, available_cities = initialize_pipeline()

# --- 3. Strict Dataset Lookups & Helper Functions ---
def get_historical_ibge(city_name: str) -> float:
    """
    Maps the cities extracted directly from the CSV to their economic multipliers 
    so the UI can correctly feed the model's feature matrix.
    """
    city_clean = city_name.lower().strip()
    indexes = {
        "são paulo": 1.50,
        "rio de janeiro": 1.45,
        "curitiba": 1.40,
        "porto alegre": 1.35,
        "belo horizonte": 1.30,
        "pelotas": 1.05
    }
    return indexes.get(city_clean, 1.0)

# Build strict Location Dropdown from the CSV column
if not available_cities:
    available_cities = ["No Historical Data Found"]

ibge_city_map = {city: get_historical_ibge(city) for city in available_cities}
ibge_city_map = dict(sorted(ibge_city_map.items()))

# Build strict Artist Dropdown from the CSV Artist column
sorted_artists = sorted(real_demand_dict.items(), key=lambda x: x[1], reverse=True)
top_artists = [artist.title() for artist, score in sorted_artists]
if not top_artists:
    top_artists = ["No Artists in Dataset"]

def calculate_operational_costs(capacity: int, production_tier: str) -> float:
    """Calculates simulated base venue operational costs."""
    base_cost_per_head = 15.00
    tier_multipliers = {
        "Low (Indie/Acoustic)": 0.8,
        "Standard": 1.0,
        "High (Arena/Stadium)": 1.6
    }
    multiplier = tier_multipliers.get(production_tier, 1.0)
    return capacity * base_cost_per_head * multiplier

# --- 4. Sidebar UI (User Inputs) ---
st.sidebar.title("Amanhã Tech")
st.sidebar.header("Historical Booking Simulator")

st.sidebar.subheader("1. Location & Venue")
selected_city = st.sidebar.selectbox("Target Market (City)", options=list(ibge_city_map.keys()), index=0)
derived_ibge = ibge_city_map[selected_city]
input_capacity = st.sidebar.slider("Venue Physical Capacity", min_value=300, max_value=80000, value=25000, step=500)

st.sidebar.subheader("2. Talent Acquisition")
input_artist = st.sidebar.selectbox("Artist / Band Name", options=top_artists)
derived_demand = real_demand_dict.get(input_artist.lower(), 0.0)

st.sidebar.subheader("3. Operational Assumptions")
production_tier = st.sidebar.selectbox("Production Requirements", ["Low (Indie/Acoustic)", "Standard", "High (Arena/Stadium)"], index=1)
bar_spend_per_head = st.sidebar.slider("Avg. F&B Spend per Attendee (R$)", min_value=0.0, max_value=150.0, value=45.0, step=5.0)

# --- 5. Machine Learning Inference & Math ---
input_data = pd.DataFrame({
    'capacity': [input_capacity],
    'spotify_demand': [derived_demand],
    'ibge_index': [derived_ibge]
})

# Execute Scikit-Learn Predictions based on the CSV trends
predicted_price = regressor.predict(input_data)[0]
sellout_probability = classifier.predict_proba(input_data)[0][1]

# Attendance Math
raw_fill = (derived_demand * 0.85) + (sellout_probability * 0.25)
expected_fill_rate = min(1.0, raw_fill)
expected_tickets_sold = int(input_capacity * expected_fill_rate)

# Financial Computations
gross_ticket_revenue = predicted_price * expected_tickets_sold
gross_fb_revenue = expected_tickets_sold * bar_spend_per_head
total_gross_revenue = gross_ticket_revenue + gross_fb_revenue

operational_costs = calculate_operational_costs(input_capacity, production_tier)
artist_fee_estimate = gross_ticket_revenue * 0.70  # Standard 70% door deal assumption
total_costs = operational_costs + artist_fee_estimate

net_profit = total_gross_revenue - total_costs
roi_percentage = (net_profit / total_costs) * 100 if total_costs > 0 else 0

# --- 6. Main Dashboard UI ---
st.title("Venue Booking Intelligence")
st.markdown("A predictive analytics fabric optimizing tour routing, ticket pricing, and venue margins through machine learning on historical datasets.")

st.header(f"Tour Analysis: {input_artist} live in {selected_city}")

# Row 1: Core Performance Metrics
st.subheader("Performance Predictions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Optimal Ticket Price", value=f"R$ {predicted_price:.2f}")
with col2:
    st.metric(label="Sell-Out Probability", value=f"{sellout_probability * 100:.1f}%")
with col3:
    st.metric(label="Expected Fill Rate", value=f"{expected_fill_rate * 100:.1f}%")
with col4:
    st.metric(label="Est. Tickets Sold", value=f"{expected_tickets_sold:,}")

st.divider()

# Row 2: Financial Metrics
st.subheader("Financial Projections")
fcol1, fcol2, fcol3, fcol4 = st.columns(4)

with fcol1:
    st.metric(label="Total Gross Revenue", value=f"R$ {total_gross_revenue:,.2f}")
with fcol2:
    st.metric(label="Estimated Fixed/Var Costs", value=f"R$ {total_costs:,.2f}")
with fcol3:
    delta_color = "normal" if net_profit > 0 else "inverse"
    st.metric(label="Projected Net Profit", value=f"R$ {net_profit:,.2f}", delta=f"{roi_percentage:.1f}% ROI", delta_color=delta_color)
with fcol4:
    st.metric(label="Food & Bev Contribution", value=f"R$ {gross_fb_revenue:,.2f}")

# --- 7. Data Visualizations (Altair) ---
st.markdown("### Operational Insights & Revenue Breakdown")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Bar Chart: Financial Flow
    fin_data = pd.DataFrame({
        "Category": ["Ticket Rev.", "F&B Rev.", "Venue Ops", "Artist Fee", "Net Profit"],
        "Amount (R$)": [gross_ticket_revenue, gross_fb_revenue, -operational_costs, -artist_fee_estimate, net_profit],
        "Ledger": ["Income", "Income", "Expense", "Expense", "Bottom Line"]
    })
    
    bar_chart = alt.Chart(fin_data).mark_bar().encode(
        x=alt.X("Category", sort=None, title="", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("Amount (R$)", title="Reais (R$)"),
        color=alt.Color(
            "Ledger", 
            scale=alt.Scale(domain=["Income", "Expense", "Bottom Line"], range=["#2ca02c", "#d62728", "#1f77b4"])
        ),
        tooltip=["Category", "Amount (R$)", "Ledger"]
    ).properties(height=350)
    
    st.altair_chart(bar_chart, use_container_width=True)

with chart_col2:
    # Donut Chart: Venue Utilization
    cap_data = pd.DataFrame({
        "Status": ["Tickets Sold", "Unsold Capacity"],
        "Count": [expected_tickets_sold, input_capacity - expected_tickets_sold]
    })
    
    donut_chart = alt.Chart(cap_data).mark_arc(innerRadius=70).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(
            field="Status", 
            type="nominal", 
            scale=alt.Scale(domain=["Tickets Sold", "Unsold Capacity"], range=["#1f77b4", "#333333"]),
            legend=alt.Legend(orient="bottom")
        ),
        tooltip=["Status", "Count"]
    ).properties(height=350)
    
    st.altair_chart(donut_chart, use_container_width=True)

st.divider()

# --- 8. Backend Explainer for Hackathon Judges ---
with st.expander("System Architecture & Raw Inferences (Pitch Documentation)"):
    st.markdown("""
    **Intelligence Pipeline Overview:**
    This platform maps historical streaming demand directly against local municipal economic multipliers.
    A `LogisticRegression` pipeline classifies the sell-out risk curve based on past performance, 
    while a `Ridge` regression model optimizes the continuous pricing variable.
    """)
    
    st.markdown("**Data Source:** `historical_concerts.csv` (Strict Enforcement)")
    
    st.code(f"""
    --- Inference Matrix ---
    Input Vector: [Capacity: {input_capacity}, Demand: {derived_demand:.4f}, IBGE: {derived_ibge:.2f}]
    
    --- Pipeline Outputs ---
    Raw Sellout Probability Output: {sellout_probability:.4f}
    Raw Optimal Ticket Price Output: {predicted_price:.4f}
    """)

st.caption(f"Amanhã Tech Alpha v2.0 | Engine Render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")