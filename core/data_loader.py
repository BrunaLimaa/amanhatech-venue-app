import pandas as pd

def fast_read_lastfm():
    # Adjust this path if your file is named differently
    file_path = "data/lastfm_1k.parquet" 
    
    columns_to_read = ['user_id', 'artist_name', 'country', 'age'] 
    
    print("Loading Last.fm data via PyArrow...")
    
    try:
        df = pd.read_parquet(file_path, engine='pyarrow', columns=columns_to_read)
        print(f"Successfully loaded {len(df):,} rows.")
        
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Total Memory Usage: {memory_mb:.2f} MB")
        print("\nSample Data:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

if __name__ == "__main__":
    df = fast_read_lastfm()