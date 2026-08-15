from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_models(X, y_price, y_sellout):
    # make_pipeline automatically scales the inputs (Capacity, Demand, IBGE)
    # so they carry equal weight in the mathematical formula.
    
    # Ridge handles the continuous ticket price prediction
    regressor = make_pipeline(StandardScaler(), Ridge(random_state=42))
    regressor.fit(X, y_price)
    
    # Logistic Regression creates a smooth 0-100% probability curve for the sellout
    classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    classifier.fit(X, y_sellout)
    
    return regressor, classifier