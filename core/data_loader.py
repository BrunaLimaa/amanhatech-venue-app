import os
import pandas as pd
import numpy as np
import requests

def fetch_jambase_venues(api_key):
    # Keep your existing Jambase logic here
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
                records.append({"capacity": capacity if capacity else 1500})
            return pd.DataFrame(records)
    except Exception:
        pass
    return pd.DataFrame()

def fetch_ibge_data():
    # Keep your existing IBGE logic here
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/RS/municipios"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
    except Exception:
        pass
    return pd.DataFrame()

def load_training_data(jambase_api_key=None):
    lastfm_path = "data/lastfm_1k.parquet"
    demand_dict = {}
    
    if os.path.exists(lastfm_path):
        df_lastfm = pd.read_parquet(
            lastfm_path, 
            engine="pyarrow", 
            columns=["user_id", "artist_name", "country"]
        )
        demand_series = df_lastfm.groupby("artist_name")["user_id"].count().reset_index()
        demand_series.rename(columns={"user_id": "spotify_demand"}, inplace=True)
        
        # Logarithmic scaling
        demand_series["spotify_demand"] = np.log1p(demand_series["spotify_demand"])
        max_demand = demand_series["spotify_demand"].max()
        demand_series["spotify_demand"] = demand_series["spotify_demand"] / (max_demand if max_demand > 0 else 1)
        
        # Extract the real data into a fast-lookup dictionary
        # Lowercase the names to make the frontend search case-insensitive
        demand_dict = dict(zip(demand_series["artist_name"].str.lower(), demand_series["spotify_demand"]))
    else:
        demand_series = pd.DataFrame({
            "artist_name": ["Fallback Artist"], 
            "spotify_demand": [0.75]
        })

    n = len(demand_series)
    
    venues_df = pd.DataFrame()
    if jambase_api_key:
        venues_df = fetch_jambase_venues(jambase_api_key)
    
    if venues_df.empty or "capacity" not in venues_df.columns:
        capacities = np.random.randint(300, 5000, size=n)
    else:
        capacities = venues_df["capacity"].sample(n=n, replace=True).values

    ibge_df = fetch_ibge_data()
    if not ibge_df.empty:
        ibge_index = np.random.uniform(0.5, 1.5, size=n)
    else:
        ibge_index = np.random.uniform(0.5, 1.5, size=n)

    X = pd.DataFrame({
        "capacity": capacities,
        "spotify_demand": demand_series["spotify_demand"].values,
        "ibge_index": ibge_index
    })

    base_price = 30.0
    prices = base_price + (X["spotify_demand"] * 120.0 * X["ibge_index"]) - (X["capacity"] * 0.002) 
    prices = prices + np.random.normal(0, 10.0, size=n)
    prices = np.clip(prices, 15.00, 500.00)

    sellout_logic = (X["spotify_demand"] * 1.5) - (X["capacity"] / 3000.0) + (X["ibge_index"] * 0.3) + np.random.normal(0, 0.25, size=n)
    sellout = np.where(sellout_logic > 1.0, 1, 0)

    # Return the dictionary alongside the ML variables
    return X, prices, sellout, demand_dict