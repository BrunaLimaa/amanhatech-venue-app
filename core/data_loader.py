import os
import pandas as pd
import numpy as np
import requests

def fetch_jambase_venues(api_key):
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
    
    if os.path.exists(lastfm_path):
        df_lastfm = pd.read_parquet(
            lastfm_path, 
            engine="pyarrow", 
            columns=["user_id", "artist_name", "country"]
        )
        demand_series = df_lastfm.groupby("artist_name")["user_id"].count().reset_index()
        demand_series.rename(columns={"user_id": "spotify_demand"}, inplace=True)
        max_demand = demand_series["spotify_demand"].max()
        demand_series["spotify_demand"] = demand_series["spotify_demand"] / (max_demand if max_demand > 0 else 1)
    else:
        demand_series = pd.DataFrame({
            "artist_name": ["Fallback Artist"], 
            "spotify_demand": [0.75]
        })

    venues_df = pd.DataFrame()
    if jambase_api_key:
        venues_df = fetch_jambase_venues(jambase_api_key)
    
    if venues_df.empty or "capacity" not in venues_df.columns:
        capacities = np.random.randint(400, 3500, size=len(demand_series))
    else:
        capacities = venues_df["capacity"].sample(n=len(demand_series), replace=True).values

    ibge_df = fetch_ibge_data()
    if not ibge_df.empty:
        ibge_index = np.random.uniform(0.7, 1.3, size=len(demand_series))
    else:
        ibge_index = np.ones(len(demand_series))

    n = len(demand_series)
    X = pd.DataFrame({
        "capacity": capacities[:n],
        "spotify_demand": demand_series["spotify_demand"].values[:n],
        "ibge_index": ibge_index[:n]
    })

    base_price = 30.0
    prices = base_price + (X["spotify_demand"] * 100.0 * X["ibge_index"]) - (X["capacity"] * 0.001)
    prices = np.clip(prices, 15.00, 400.00)

    sellout_logic = (X["spotify_demand"] * 1.3) - (X["capacity"] / 5000.0)
    sellout = np.where(sellout_logic > 0.4, 1, 0)

    return X, prices, sellout