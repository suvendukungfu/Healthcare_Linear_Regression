import pandas as pd
from sklearn.linear_model import LinearRegression

def load_data(path="data/healthcare_data.csv"):
    """Load healthcare dataset"""
    return pd.read_csv(path)

def train_model(df):
    """Train Linear Regression model"""
    X = df.drop("RiskScore", axis=1)
    y = df["RiskScore"]

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict_risk(model, patient_data):
    """Predict healthcare risk score"""
    return model.predict(patient_data)
