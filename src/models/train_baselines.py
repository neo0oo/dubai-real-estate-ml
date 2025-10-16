import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.features.make_features import build_features, train_val_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score

def add_engineered_features(df):
    # Basic ratios
    df['area_per_bed'] = df['area_in_sqft'] / df['beds']
    df['area_per_bath'] = df['area_in_sqft'] / df['baths']
    df['beds_per_bath'] = df['beds'] / df['baths']
    
    # Price-based features
    df['price_per_sqft'] = df['rent'] / df['area_in_sqft']
    
    # Location-based features
    location_avg_price = df.groupby('location')['rent'].mean()
    df['location_avg_price'] = df['location'].map(location_avg_price)
    
    # Additional features
    df['total_rooms'] = df['beds'] + df['baths']
    df['area_per_total_rooms'] = df['area_in_sqft'] / df['total_rooms']
    df['price_per_room'] = df['rent'] / df['total_rooms']
    
    # Type-based features
    type_avg_price = df.groupby('type')['rent'].mean()
    df['type_avg_price'] = df['type'].map(type_avg_price)
    
    # Interaction features
    df['area_per_bed_bath'] = df['area_in_sqft'] / (df['beds'] * df['baths'])
    
    return df

def select_important_features(x, y, k=10):
    """Select k most important features and return their names"""
    selector = SelectKBest(f_regression, k=k)
    selector.fit(x, y)
    feature_mask = selector.get_support()
    selected_features = x.columns[feature_mask].tolist()
    return selected_features

def optimize_random_forest():
    n_estimators = [100, 200, 300]
    max_depth = [10, 12, 14]
    min_samples_leaf = [5, 10, 15]
    max_features = ["sqrt", "log2"]
    n_jobs = [-1, 1]
    
    best_model = None
    best_score = -np.inf
    
    for n_estimators in n_estimators:
        for max_depth in max_depth:
            for min_samples_leaf in min_samples_leaf:
                for max_features in max_features:
                    for n_jobs in n_jobs:
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                    min_samples_leaf=min_samples_leaf, max_features=max_features,
                                                    n_jobs=n_jobs)
                        score = model.score(x_train, y_train)
                        if score > best_score:
                            best_score = score
                            best_model = model
    return best_model

def evaluate_model(model, X, y):
    cv_scores = cross_val_score(
        model, 
        X, 
        y, 
        cv=5, 
        scoring='r2',
        n_jobs=-1
    )
    return cv_scores.mean(), cv_scores.std()

df = pd.read_csv("data/processed/cleaned_dubai_properties.csv")
df = add_engineered_features(df)
x, y = build_features(df, encoding="label")


important_features = select_important_features(x, y)

x = x[important_features]

x_train, x_val, y_train, y_val = train_val_split(x, y)

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
    print(f"\nTraining {name}...")
    model = log_wrap(base_model) if USE_LOG_WRAPPER else base_model
    
    r2_mean, r2_std = evaluate_model(model, x, y)
    print(f"{name} CV R² Score: {r2_mean:.3f} (+/- {r2_std:.3f})")
    
    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"{name} -> MAE: {mae:,.0f} AED, RMSE: {rmse:,.0f} AED, R²: {r2:.3f}")
