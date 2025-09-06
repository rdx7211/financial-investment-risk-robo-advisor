import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


df = pd.read_pickle(r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\data\processed\processed_data.pkl")
df

df = df.replace([np.inf, -np.inf], np.nan).dropna()

def train_and_save_best_model(df, model_path):
    """
    Trains multiple models, evaluates them, selects the best model, and saves it.
    """
    X = df.drop('RISK_TOLERANCE', axis=1)
    y = df['RISK_TOLERANCE']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBRegressor": XGBRegressor(),
        "AdaBoost Regressor": AdaBoostRegressor(),
    }
    
    best_model = None
    best_r2 = -np.inf
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    if best_model:
        joblib.dump(best_model, model_path)
        print(f"Best model saved: {model_path}")


train_and_save_best_model(df,model_path=r"D:\Investor_Risk_Tolerance_and_Robo-Advisor\models\best_model.pkl")