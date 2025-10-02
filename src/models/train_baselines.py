import pandas as pd 
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from src.features.make_features import build_features, train_val_split

df = pd.read_csv("data/processed/cleaned_dubai_properties.csv")
x,y = build_features(df,encoding="onehot")
x_train, x_val, y_train, y_val = train_val_split(x,y)
models = {
    "Dummy": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_depth = 8, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")