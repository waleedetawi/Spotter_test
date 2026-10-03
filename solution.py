import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# Gracefully handle optional gradient boosting libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


df_train = pd.read_csv('train-test.csv')
df_val = pd.read_csv('validation.csv')
df_december = pd.read_csv('december-chart-inputs.csv')
print(f'Training data: {df_train.shape}')
df_train.head()

df_train.isna().sum()

(df_train['weight']<0).sum()


df_train = df_train[df_train['distance'] > 0] # checked that all of them are postive 
df_train['weight'] = df_train['weight'].abs() # since we had negative weights which doesnt make sense 
print('Cleaning complete.')

# Show all rows where the 'weight' column is missing
missing_weights_df = df_train[df_train['weight'].isnull()]

print(f"Total rows with missing weights: {len(missing_weights_df)}")
missing_weights_df

plt.figure(figsize=(10, 6))
# Using seaborn to plot the distribution of weights, ignoring NaNs
sns.histplot(df_train['weight'].dropna(), bins=50, kde=True, color='blue')

plt.title('Distribution of Load Weights', fontsize=14)
plt.xlabel('Weight', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Print basic statistics for the weight column
print("Weight Summary Statistics:")
print(df_train['weight'].describe())


WEIGHT_MEDIAN = df_train['weight'].median()
df_train['weight'] = df_train['weight'].fillna(WEIGHT_MEDIAN)

# Show rows with missing market_index
missing_market_df = df_train[df_train['market_index'].isnull()]
print(f"Total rows with missing market_index: {len(missing_market_df)}")
missing_market_df.head()


plt.figure(figsize=(10, 6))
# Plot the distribution of market_index, ignoring NaNs
sns.histplot(df_train['market_index'].dropna(), bins=50, kde=True, color='green')

plt.title('Distribution of Market Index', fontsize=14)
plt.xlabel('Market Index', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Print basic statistics for the market_index column
print("Market Index Summary Statistics:")
print(df_train['market_index'].describe())


# Group by date to see how many missing market indices there are per day
missing_by_date = df_train[df_train['market_index'].isnull()]['date'].value_counts()

print(f"Number of unique dates with missing market indices: {len(missing_by_date)}")
print("\nTop 10 dates with the most missing market indices:")
print(missing_by_date.head(10))


# 1. First, calculate the average market index for the specific Date + Pickup City
city_daily_market = df_train.groupby(['date', 'pickup'])['market_index'].transform('mean')
df_train['market_index'] = df_train['market_index'].fillna(city_daily_market)

daily_market = df_train.groupby('date')['market_index'].transform('mean')
df_train['market_index'] = df_train['market_index'].fillna(daily_market)


df_train.isnull().sum()

def engineer_dates(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df
df_train = engineer_dates(df_train)

def engineer_geography(df):
    df = df.copy()
    df['lat_diff'] = df['delivery_lat'] - df['pickup_lat']
    df['lon_diff'] = df['delivery_lon'] - df['pickup_lon']
    df['geo_distance'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)
    df['log_distance'] = np.log1p(df['distance'])
    return df
df_train = engineer_geography(df_train)

def engineer_interactions(df):
    df = df.copy()
    df['rate_per_mile_signal'] = df['quote_signal'] * df['distance']
    df['distance_x_weight'] = df['distance'] * df['weight']
    df['distance_x_signal'] = df['distance'] * df['quote_signal']
    return df
df_train = engineer_interactions(df_train)

df_train

le_pickup = LabelEncoder()
le_delivery = LabelEncoder()
le_equipment = LabelEncoder()
def encode_categoricals(df, fit=False):
    df = df.copy()
    if fit:
        df['pickup_enc'] = le_pickup.fit_transform(df['pickup'])
        df['delivery_enc'] = le_delivery.fit_transform(df['delivery'])
        df['equipment_enc'] = le_equipment.fit_transform(df['equipment'])
    else:
        df['pickup_enc'] = df['pickup'].map(lambda x: le_pickup.transform([x])[0] if x in le_pickup.classes_ else -1)
        df['delivery_enc'] = df['delivery'].map(lambda x: le_delivery.transform([x])[0] if x in le_delivery.classes_ else -1)
        df['equipment_enc'] = df['equipment'].map(lambda x: le_equipment.transform([x])[0] if x in le_equipment.classes_ else -1)
    return df
df_train = encode_categoricals(df_train, fit=True)

feature_cols = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance', 'weight', 'market_index', 'quote_signal', 'day_of_week', 'month', 'day_sin', 'day_cos', 'lat_diff', 'lon_diff', 'geo_distance', 'log_distance', 'rate_per_mile_signal', 'distance_x_weight', 'distance_x_signal', 'pickup_enc', 'delivery_enc', 'equipment_enc']
X = df_train[feature_cols].values
y = df_train['posted_rate'].values
X = np.nan_to_num(X, nan=0.0)
print(f'Feature matrix shape: {X.shape}')

dates = df_train['date'].values
sorted_idx = np.argsort(dates)
split_point = int(len(sorted_idx) * 0.8)
train_idx, test_idx = sorted_idx[:split_point], sorted_idx[split_point:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f'Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}')


models_and_params = {
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 7],
            "learning_rate": [0.05, 0.1]
        }
    }
}

if 'XGBRegressor' in globals():
    models_and_params["XGBoost"] = {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    }

if 'LGBMRegressor' in globals():
    models_and_params["LightGBM"] = {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    }

best_overall_model = None
best_overall_mae = float('inf')
best_overall_name = ""
tuned_results = {}

print("🚀 Starting Hyperparameter Tuning for Models...\n")

for name, mp in models_and_params.items():
    print(f"[{name}] Starting tuning...")
    start_time = time.time()
    
    search = RandomizedSearchCV(
        mp["model"], 
        param_distributions=mp["params"], 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42, 
        n_jobs=1 
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    elapsed_time = time.time() - start_time
    
    tuned_results[name] = {"model": best_model, "mae": mae, "params": search.best_params_}
    
    print(f"[{name}] ✅ Done in {elapsed_time:.1f}s | Test MAE: ${mae:.2f}")
    print(f"[{name}] Best Params: {search.best_params_}\n")
    
    if mae < best_overall_mae:
        best_overall_mae = mae
        best_overall_model = best_model
        best_overall_name = name

print("="*60)
print(f"🏆 WINNING MODEL: {best_overall_name}")
print(f"🏆 WINNING MAE: ${best_overall_mae:.2f}")
print("="*60)

final_model = best_overall_model
print("Retraining best model on full dataset...")
final_model.fit(X, y)
print("Done.")

# Recalculate medians directly from the training dataset
train_market_median = df_train["market_index"].median()
train_signal_median = df_train["quote_signal"].median()

df_dec_clean = df_december.copy()

# Fill fixed values for Lexington -> Fort Wayne using exact coordinates from training
df_dec_clean["pickup_lat"] = 36.99152
df_dec_clean["pickup_lon"] = -84.99876
df_dec_clean["delivery_lat"] = 41.31561
df_dec_clean["delivery_lon"] = -85.36206
df_dec_clean["market_index"] = train_market_median
df_dec_clean["quote_signal"] = train_signal_median

df_dec_clean["pickup"] = "Lexington"
df_dec_clean["delivery"] = "Fort Wayne"

# We must ensure equipment and weight match the test requirement
df_dec_clean["equipment"] = "Dry Van"
df_dec_clean["weight"] = 32000.0

# Engineering
df_dec_clean = engineer_dates(df_dec_clean)
df_dec_clean = engineer_geography(df_dec_clean)
df_dec_clean = engineer_interactions(df_dec_clean)
df_dec_clean = encode_categoricals(df_dec_clean, fit=False)

X_dec = df_dec_clean[feature_cols].values
X_dec = np.nan_to_num(X_dec, nan=0.0)

dec_preds = final_model.predict(X_dec)
dec_preds = np.maximum(dec_preds, 1.0)

df_december["predicted_rate"] = dec_preds
df_december.to_csv("december-chart-inputs.csv", index=False)
print("Saved december-chart-inputs.csv")

import os
print('\nRunning Scorer...')
os.system('python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv')
