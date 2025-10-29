# All Imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Set Plot Style
sns.set_style('whitegrid')
print("Libraries imported successfully!")


# Load Dataset 
try:
    df = pd.read_csv('honey_purity_dataset.csv')
    print("✅ 'honey_purity_dataset.csv' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'honey_purity_dataset.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this notebook.")


if 'df' in locals():
    # Display the first 5 rows
    print("\nData Head:")
    print(df.head())

    # Display data types and non-null values
    print("\nData Info:")
    df.info()

    # Display summary statistics for numerical columns
    print("\nData Description:")
    print(df.describe())

    # --- Correlation Matrix Visualization ---
    print("\nCorrelation Matrix Heatmap:")
    plt.figure(figsize=(12, 10))
    # Ensure only numeric columns are used for correlation
    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Honey Features')
    plt.show()


# Selete input values and output values
TARGET = 'Price'
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# One-Hot Encode the 'Pollen_analysis' column
# This converts the text-based honey types into a numerical format for the model.
X = pd.get_dummies(X, columns=['Pollen_analysis'], drop_first=True)

print("Feature set shape after one-hot encoding:", X.shape)
print("Example of features after encoding:")
print(X.head())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset info
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create Model
# Define the models to be trained
models = {
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("--- Model Training and Evaluation ---")
best_model = None
best_r2 = -1
best_model_name = ""

results = []

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({'Model': name, 'R-squared': r2, 'MAE': mae})

    # Keep track of the best model based on R-squared
    if 98 > r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Print Results
results_df = pd.DataFrame(results)
print("\n--- Evaluation Results ---")
print(results_df.sort_values(by='R-squared', ascending=False))

print(f"\n🏆 Best performing model is '{best_model_name}' with an R-squared of {best_r2:.4f}")

# Save the model
joblib.dump(best_model, 'best_model.pkl')
print(f"✅ Best model ('{best_model_name}') saved to 'best_model.pkl'")

# Save the Column Order
# This is CRUCIAL for ensuring the app's input matches the model's expectations.
model_columns = X.columns
joblib.dump(model_columns, 'model_columns.pkl')
print(f"✅ Model columns saved to 'model_columns.pkl'")

# Save the List of Pollen Types for the App's Dropdown Menu
# This makes the app adaptable to the specific honey types in your dataset.
pollen_types_list = sorted(df['Pollen_analysis'].unique().tolist())
joblib.dump(pollen_types_list, 'pollen_types.pkl')
print(f"✅ List of {len(pollen_types_list)} pollen types saved to 'pollen_types.pkl'")

# Print plots
# 1. Actual vs Predicted Honey Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='dodgerblue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Honey Prices')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Distribution of Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='coral')
plt.title('Distribution of Residuals')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()
