import h5py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV  # For hyperparameter tuning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from scipy.stats import randint, uniform
import time

file_path = 'C:/Users/muham/Downloads/SMAP_L2_SM_SP_1AIWDV_20240927T145143_20240928T020009_120W38N_R19240_003.h5'

# Northern California Boundaries
north_lat = 42.0
south_lat = 37.0
west_lon = -125.0
east_lon = -120.0

# Load data from HDF5 file
with h5py.File(file_path, 'r') as h5file:
    soil_moisture_1km = h5file['Soil_Moisture_Retrieval_Data_1km']['soil_moisture_1km'][:]
    latitude_1km = h5file['Soil_Moisture_Retrieval_Data_1km']['latitude_1km'][:]
    longitude_1km = h5file['Soil_Moisture_Retrieval_Data_1km']['longitude_1km'][:]

# Handle Missing Data
missing_value = -9999.0
soil_moisture_1km[np.isclose(soil_moisture_1km, missing_value)] = np.nan
soil_moisture_1km_imputed = np.nan_to_num(soil_moisture_1km, nan=0)
soil_moisture_1km_imputed *= 100  # Scale values

# Create a mask for the defined geographic boundaries
longitude_grid_1km, latitude_grid_1km = np.meshgrid(longitude_1km[0, :], latitude_1km[:, 0])
mask_1km = (latitude_1km[:, 0] >= south_lat) & (latitude_1km[:, 0] <= north_lat)
mask_1km = mask_1km[:, np.newaxis] & ((longitude_1km[0, :] >= west_lon) & (longitude_1km[0, :] <= east_lon))
masked_soil_moisture_1km = np.ma.masked_where(~mask_1km, soil_moisture_1km_imputed)

# Segment Data Function
def create_segments(data, segment_size, overlap):
    n_rows, n_cols = data.shape
    segments = []
    for i in range(0, n_rows - segment_size + 1, overlap):
        for j in range(0, n_cols - segment_size + 1, overlap):
            segment = data[i:i + segment_size, j:j + segment_size].copy()
            segment_mean = np.nanmean(segment)
            segment[np.isnan(segment)] = segment_mean
            segments.append(segment)
    return np.array(segments)

# Segment the Imputed Data
segment_size = 10
overlap = 5
segments = create_segments(soil_moisture_1km_imputed, segment_size, overlap)

# Calculate Labels
labels = np.mean(segments, axis=(1, 2))

# Flatten the Segments for Machine Learning
segments_flat = segments.reshape(segments.shape[0], -1)

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(segments_flat, labels, test_size=0.2, random_state=42)

# 70/30 Split of the Training Data
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Scale the Features
min_max_scaler = MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train_final)
X_val_scaled = min_max_scaler.transform(X_val)
X_test_scaled = min_max_scaler.transform(X_test)

# Print Shapes of Training, Validation, and Test Sets
print("X_train_final shape:", X_train_final.shape)
print("y_train_final shape:", y_train_final.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Model Training
def evaluate_model(model, X, y_true, model_name):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} R-squared: {r2:.4f}")
    print(f"{model_name} MSE: {mse:.4f}")
    print(f"{model_name} RMSE: {rmse:.4f}")
    return r2, mse, rmse

# Random Forest
start_time = time.time()
print("Training Random Forest...")
param_dist_rf = {
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=20,
                                      cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search_rf.fit(X_train_scaled, y_train_final)
rf_model = random_search_rf.best_estimator_
end_time = time.time()
print(f"Random Forest training completed in {end_time - start_time:.2f} seconds.")
print(f"Best Random Forest hyperparameters: {random_search_rf.best_params_}")

# CatBoost
start_time = time.time()
print("\nTraining CatBoost...")
param_dist_cat = {
    'iterations': randint(200, 1000),
    'learning_rate': uniform(0.01, 0.1),
    'depth': randint(4, 10),
    'l2_leaf_reg': uniform(1, 10),
    'random_strength': uniform(0.5, 1.5),
    'bagging_temperature': uniform(0, 1)
}
cat_model = CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=0)
random_search_cat = RandomizedSearchCV(cat_model, param_distributions=param_dist_cat, n_iter=10,
                                       cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search_cat.fit(X_train_scaled, y_train_final)
cat_model = random_search_cat.best_estimator_
end_time = time.time()
print(f"CatBoost training completed in {end_time - start_time:.2f} seconds.")
print(f"Best CatBoost hyperparameters: {random_search_cat.best_params_}")

# XGBoost
start_time = time.time()
print("\nTraining XGBoost...")
param_dist_xgb = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 8),
    'reg_lambda': uniform(0.1, 10),
    'reg_alpha': uniform(0, 1)
}
xgb_model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
random_search_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=10,
                                       cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search_xgb.fit(X_train_scaled, y_train_final)
xgb_model = random_search_xgb.best_estimator_
end_time = time.time()
print(f"XGBoost training completed in {end_time - start_time:.2f} seconds.")
print(f"Best XGBoost hyperparameters: {random_search_xgb.best_params_}")

# Evaluation
print("\nEvaluating models on validation set:")
r2_rf, mse_rf, rmse_rf = evaluate_model(rf_model, X_val_scaled, y_val, "Random Forest")
r2_cat, mse_cat, rmse_cat = evaluate_model(cat_model, X_val_scaled, y_val, "CatBoost")
r2_xgb, mse_xgb, rmse_xgb = evaluate_model(xgb_model, X_val_scaled, y_val, "XGBoost")

# Custom Accuracy Function
def calculate_regression_accuracy(y_true, y_pred, tolerance=0.1):
    differences = np.abs(y_true - y_pred)
    tolerance_values = np.abs(y_true) * tolerance
    within_tolerance = differences <= tolerance_values
    accuracy_percentage = np.mean(within_tolerance) * 100
    return accuracy_percentage

# Calculate accuracy for validation set
print("\nCalculating Custom Accuracy for Validation Set...")
accuracy_rf = calculate_regression_accuracy(y_val, rf_model.predict(X_val_scaled))
accuracy_cat = calculate_regression_accuracy(y_val, cat_model.predict(X_val_scaled))
accuracy_xgb = calculate_regression_accuracy(y_val, xgb_model.predict(X_val_scaled))

print(f"Random Forest Custom Accuracy: {accuracy_rf:.2f}%")
print(f"CatBoost Custom Accuracy: {accuracy_cat:.2f}%")
print(f"XGBoost Custom Accuracy: {accuracy_xgb:.2f}%")

# Evaluate on Test Set
print("\nEvaluating models on test set:")
r2_rf_test, mse_rf_test, rmse_rf_test = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
r2_cat_test, mse_cat_test, rmse_cat_test = evaluate_model(cat_model, X_test_scaled, y_test, "CatBoost")
r2_xgb_test, mse_xgb_test, rmse_xgb_test = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")

# Custom Accuracy for Test Set
print("\nCalculating Custom Accuracy for Test Set...")
accuracy_rf_test = calculate_regression_accuracy(y_test, rf_model.predict(X_test_scaled))
accuracy_cat_test = calculate_regression_accuracy(y_test, cat_model.predict(X_test_scaled))
accuracy_xgb_test = calculate_regression_accuracy(y_test, xgb_model.predict(X_test_scaled))

print(f"Random Forest Custom Accuracy on Test Set: {accuracy_rf_test:.2f}%")
print(f"CatBoost Custom Accuracy on Test Set: {accuracy_cat_test:.2f}%")
print(f"XGBoost Custom Accuracy on Test Set: {accuracy_xgb_test:.2f}%")

# Calculate average predictions from all models
y_pred_avg = (rf_model.predict(X_test_scaled) +
              cat_model.predict(X_test_scaled) +
              xgb_model.predict(X_test_scaled)) / 3

# Evaluate average predictions
print("\nEvaluating Average Predictions on Test Set:")
r2_avg = r2_score(y_test, y_pred_avg)
mse_avg = mean_squared_error(y_test, y_pred_avg)
rmse_avg = np.sqrt(mse_avg)
print(f"Average Predictions R-squared: {r2_avg:.4f}")
print(f"Average Predictions MSE: {mse_avg:.4f}")
print(f"Average Predictions RMSE: {rmse_avg:.4f}")

# Custom Accuracy for Average Predictions
accuracy_avg = calculate_regression_accuracy(y_test, y_pred_avg)
print(f"Average Predictions Custom Accuracy on Test Set: {accuracy_avg:.2f}%")

# Optional: Show an output using the average predictions
sample_index = 0  # Change this index to see predictions for different samples
print(f"True value: {y_test[sample_index]}, Predicted Average: {y_pred_avg[sample_index]}")

import os
import joblib

# Create a directory to save the models if it does not exist
model_directory = 'E:/PRACTICE/python/pythonProject/models/'
os.makedirs(model_directory, exist_ok=True)  # Create directory if it doesn't exist

# Save the trained models
joblib.dump(rf_model, os.path.join(model_directory, 'rf_model.pkl'))
joblib.dump(cat_model, os.path.join(model_directory, 'cat_model.pkl'))
joblib.dump(xgb_model, os.path.join(model_directory, 'xgb_model.pkl'))

print("Models saved successfully!")
