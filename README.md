🏙️ Dubai Real Estate ML Project

Predicting rental prices in Dubai’s housing market using machine learning.
This project explores property data from Kaggle, applying data cleaning, feature engineering, and regression modeling (Linear, Random Forest, and HistGradientBoosting) to understand and forecast property rents.

Built with Python, Pandas, Scikit-learn, and Jupyter Notebook, the workflow follows a modular structure for clarity, scalability, and reproducibility.

🧠 Key Highlights

🧹 Cleaned and prepared real-world property data

🧩 Engineered domain-specific features (area, price ratios, location averages, etc.)

🤖 Trained multiple ML models with log-transformed targets for improved stability

📈 Evaluated models using MAE, RMSE, and R²

💾 Saved trained models, metrics, and results for reproducible analysis

📂 Project Structure
dubai-real-estate-ml/
│
├── data/
│   ├── raw/                # Original Kaggle dataset
│   ├── processed/          # Cleaned dataset ready for ML
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis (EDA)
│
├── src/
│   ├── data/
│   │   ├── load_data.py      # Load raw data
│   │   ├── clean_data.py     # Clean and preprocess data
│   │
│   ├── features/
│   │   ├── make_features.py  # Build and encode features
│   │
│   ├── models/
│   │   ├── train_baselines.py  # Train and evaluate ML models
│   │   ├── save_results.py     # Save model metrics and results
│
├── artifacts/
│   ├── metrics/              # Model performance scores
│   ├── models/               # Saved trained models
│
├── reports/
│   ├── figures/              # Visualizations and plots
│
├── results/                  # Final results and summaries
│
└── README.md
