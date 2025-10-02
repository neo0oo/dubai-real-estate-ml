import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features.make_features import build_features, train_val_split

df = pd.read_csv("data/processed/cleaned_dubai_properties.csv")
x,y = build_features(df,encoding="label")
x_train, x_val, y_train, y_val = train_val_split(x,y)

def log_wrap(model):
    return TransformedTargetRegressor(
        regressor = model,
        func = np.log1p,
        inverse_func=np.expm1,
        check_inverse = False
    )

models = {
    "Dummy": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_depth = 8, random_state=42,learning_rate=0.1,max_iter=300),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42,min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1),
}

USE_LOG_WRAPPER = True

for name, base_model in models.items():
    model = log_wrap(base_model) if USE_LOG_WRAPPER else base_model
    model.fit(x_train, y_train)
    preds = model.predict(x_val)

    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2   = r2_score(y_val, preds)
    print(f"{name} -> MAE: {mae:,.0f} AED, RMSE: {rmse:,.0f} AED, R²: {r2:.3f}")