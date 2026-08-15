import os
import pandas as pd
import numpy as np
import requests

def fetch_jambase_venues(api_key):
    """
    Fetches real venue capacities and city locations from the JamBase API.
    """
    url = "https://api.data.jambase.com/v3/venues"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"geoCountryIso2": "BR", "perPage": 100}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            venues = data.get("venues", [])
            records = []
            for v in venues:
                capacity = v.get("capacity", 1500)
                # Extract the city to dynamically populate the UI dropdown
                address = v.get("address", {})
                city = address.get("addressLocality", "")
                
                records.append({
                    "capacity": capacity if capacity else 1500,
                    "city": city.strip() if city else "Unknown"
                })
            return pd.DataFrame(records)
    except Exception as e:
        print(f"JamBase API Error: {e}")
    return pd.DataFrame()

def load_training_data(jambase_api_key=None):
    """
    Core data ingestion pipeline.
    Parses Parquet, queries APIs, and generates the synthetic financial targets.
    """
    lastfm_path = "data/lastfm_1k.parquet"
    demand_dict = {}
    
    # 1. Last.fm Parquet Ingestion
    if os.path.exists(lastfm_path):
        df_lastfm = pd.read_parquet(
            lastfm_path, 
            engine="pyarrow", 
            columns=["user_id", "artist_name"]
        )
        demand_series = df_lastfm.groupby("artist_name")["user_id"].count().reset_index()
        demand_series.rename(columns={"user_id": "spotify_demand"}, inplace=True)
        
        # Logarithmic scaling to compress massive listener counts into a smooth curve
        demand_series["spotify_demand"] = np.log1p(demand_series["spotify_demand"])
        max_demand = demand_series["spotify_demand"].max()
        demand_series["spotify_demand"] = demand_series["spotify_demand"] / (max_demand if max_demand > 0 else 1)
        
        # Build the fast-lookup dictionary for the Streamlit UI
        demand_dict = dict(zip(demand_series["artist_name"].str.lower(), demand_series["spotify_demand"]))
    else:
        # Fallback if the Parquet file is missing from the data/ folder
        demand_series = pd.DataFrame({"artist_name": ["Fallback Artist"], "spotify_demand": [0.75]})

    n = len(demand_series)
    
    # 2. JamBase API Ingestion
    venues_df = pd.DataFrame()
    if jambase_api_key:
        venues_df = fetch_jambase_venues(jambase_api_key)
    
    # Extract cities for the frontend
    jambase_cities = ["São Paulo", "Rio de Janeiro", "Porto Alegre", "Curitiba", "Belo Horizonte"]
    if not venues_df.empty and "city" in venues_df.columns:
        fetched_cities = venues_df["city"].dropna().unique().tolist()
        fetched_cities = [c for c in fetched_cities if c and c != "Unknown"]
        if fetched_cities:
            jambase_cities = fetched_cities
            
    # Assign capacities to the training matrix
    if venues_df.empty or "capacity" not in venues_df.columns:
        capacities = np.random.randint(300, 5000, size=n)
    else:
        capacities = venues_df["capacity"].sample(n=n, replace=True).values

    # 3. Target Variable Generation (The Financial Mock)
    ibge_index = np.random.uniform(0.5, 1.5, size=n)

    X = pd.DataFrame({
        "capacity": capacities,
        "spotify_demand": demand_series["spotify_demand"].values,
        "ibge_index": ibge_index
    })

    # Linear math for pricing
    base_price = 30.0
    prices = base_price + (X["spotify_demand"] * 120.0 * X["ibge_index"]) - (X["capacity"] * 0.002) 
    prices = prices + np.random.normal(0, 10.0, size=n)
    prices = np.clip(prices, 15.00, 500.00)

    # Sigmoid-friendly logic for sellouts
    sellout_logic = (X["spotify_demand"] * 1.5) - (X["capacity"] / 3000.0) + (X["ibge_index"] * 0.3) + np.random.normal(0, 0.25, size=n)
    sellout = np.where(sellout_logic > 1.0, 1, 0)

    # Return exactly 5 variables to the UI
    return X, prices, sellout, demand_dict, jambase_cities