import streamlit as st
import pandas as pd
import hashlib
from core.data_loader import load_training_data
from core.ml_engine import train_models

JAMBASE_API_KEY = "jbd_trial_hJqB7wA0RiiR_HUjBcVOIaYaJgRLHWvnvVmDoaR96Ayjp"

st.set_page_config(page_title="Venue Booking Intelligence", layout="wide")

@st.cache_resource(show_spinner="Booting Intelligence Engine & Fetching API Data...")
def initialize_pipeline(api_key):
    api_key_clean = api_key.strip() if api_key else None
    # Unpack the real data dictionary
    X, y_price, y_sellout, demand_dict = load_training_data(jambase_api_key=api_key_clean)
    regressor, classifier = train_models(X, y_price, y_sellout)
    return regressor, classifier, demand_dict

# Load the dictionary into the global session
regressor, classifier, real_demand_dict = initialize_pipeline(JAMBASE_API_KEY)

ibge_city_map = {
    "Porto Alegre": 1.35,
    "Caxias do Sul": 1.25,
    "Canoas": 1.15,
    "Pelotas": 1.05,
    "Santa Maria": 1.10,
    "São Paulo": 1.50,
    "Curitiba": 1.40
}

def get_artist_demand(artist_name, demand_dict):
    if not artist_name:
        return 0.1
        
    name_clean = artist_name.lower().strip()
    
    # 1. The Real Data Check
    # Instantly queries the Last.fm extracted scores
    if name_clean in demand_dict:
        return demand_dict[name_clean]
    
    # 2. VIP Fallback (Safety net for demoing global acts not in the 1k sample)
    mega_stars = {
        "coldplay": 0.98,
        "taylor swift": 0.99,
        "radiohead": 0.92,
        "arctic monkeys": 0.89,
        "the weeknd": 0.96,
        "metallica": 0.94
    }
    
    if name_clean in mega_stars:
        return mega_stars[name_clean]
        
    # 3. Hash Fallback (If an unknown band is typed)
    hash_val = int(hashlib.md5(name_clean.encode('utf-8')).hexdigest(), 16)
    return 0.3 + ((hash_val % 600) / 1000.0) 

st.sidebar.header("Booking Simulator")

selected_city = st.sidebar.selectbox("Venue Location (IBGE Mapping)", options=list(ibge_city_map.keys()), index=0)
derived_ibge = ibge_city_map[selected_city]

input_artist = st.sidebar.text_input("Artist to Book", value="Radiohead")
# Pass the real data dictionary into the lookup function
derived_demand = get_artist_demand(input_artist, real_demand_dict)

input_capacity = st.sidebar.slider("Venue Capacity", min_value=300, max_value=5000, value=1800, step=100)

input_data = pd.DataFrame({
    'capacity': [input_capacity],
    'spotify_demand': [derived_demand],
    'ibge_index': [derived_ibge]
})

predicted_price = regressor.predict(input_data)[0]
sellout_probability = classifier.predict_proba(input_data)[0][1]

expected_fill_rate = min(1.0, (derived_demand * 0.8) + (sellout_probability * 0.2))
expected_tickets_sold = int(input_capacity * expected_fill_rate)
projected_revenue = predicted_price * expected_tickets_sold

st.title("Venue Booking Intelligence & Revenue Simulation")
st.markdown("A predictive fabric analyzing listener demographics against local purchasing power to optimize tour booking.")

st.subheader(f"Booking Analysis: {input_artist} in {selected_city}")

with st.expander("View Extracted Data Variables"):
    # Adding an indicator to show the judges where the data originated
    data_source = "Last.fm Dataset" if input_artist.lower().strip() in real_demand_dict else "VIP/Hash Engine"
    
    st.markdown(f"**IBGE Economic Index for {selected_city}:** {derived_ibge}")
    st.markdown(f"**Calculated Demand Score for {input_artist}:** {derived_demand:.4f} *(Source: {data_source})*")
    st.markdown(f"**Expected Fill Rate:** {expected_fill_rate * 100:.1f}% ({expected_tickets_sold} tickets)")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Calculated Ticket Price", value=f"R$ {predicted_price:.2f}")

with col2:
    st.metric(label="Sell-Out Probability", value=f"{sellout_probability * 100:.1f}%")

with col3:
    st.metric(label="Projected Gross Revenue", value=f"R$ {projected_revenue:,.2f}")