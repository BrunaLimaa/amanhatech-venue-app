from core.data_loader import load_training_data

def run_loader_test():
    print("Initializing Data Loader Test...")
    
    try:
        # Execute the function from your core module
        X, prices, sellout = load_training_data()
        
        print("Success: Data loaded and transformed.")
        print(f"Feature Matrix (X) Shape: {X.shape}")
        print(f"Target Array (prices) Shape: {prices.shape}")
        print(f"Target Array (sellout) Shape: {sellout.shape}")
        
        print("\nSample Feature Data (X):")
        print(X.head())
        
        print("\nSample Target Data (prices):")
        print(prices[:5])
        
    except Exception as e:
        print(f"Error executing load_training_data: {e}")

if __name__ == "__main__":
    run_loader_test()