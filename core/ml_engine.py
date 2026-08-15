from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_models(X, y_price, y_sellout):
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    regressor.fit(X, y_price)
    
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(X, y_sellout)
    
    return regressor, classifier