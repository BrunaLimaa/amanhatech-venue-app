from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_models(X, y_price, y_sellout):
    """
    Trains the machine learning pipelines using historical box office data.
    
    X: Features DataFrame (capacity, spotify_demand, ibge_index)
    y_price: Continuous target (actual_ticket_price_brl)
    y_sellout: Binary target (sold_out)
    """
    
    # Ridge Regression for Ticket Price Optimization
    # L2 Regularization prevents overfitting to outlier stadium shows
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    regressor.fit(X, y_price)
    
    # Logistic Regression for Sell-out Probability
    # Generates a continuous 0.0 to 1.0 probability curve based on historical fill rates
    classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    classifier.fit(X, y_sellout)
    
    return regressor, classifier