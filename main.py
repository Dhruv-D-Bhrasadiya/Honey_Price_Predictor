# All Imports

import os
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
    
    # Create static/plots directory if it doesn't exist
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    plt.savefig('static/plots/correlation_matrix.png', bbox_inches='tight')
    print("✅ Correlation matrix saved to 'static/plots/correlation_matrix.png'")
    plt.close() # Close the plot window

    # --- Generate and Save EDA Plots ---
    print("\n--- Generating EDA plots for the dataset ---")
    
    # 1. Histograms of all numerical features
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(numeric_df.columns):
        plt.subplot(4, 3, i + 1)
        sns.histplot(numeric_df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}', fontsize=10)
        plt.xlabel('')
        plt.ylabel('')
    plt.tight_layout()
    plt.savefig('static/plots/feature_distributions.png')
    print("✅ Feature distribution plots saved to 'static/plots/feature_distributions.png'")
    plt.close()

    # 2. Scatter plots of key features vs. Price
    key_features = ['Purity', 'Viscosity', 'CS', 'WC']
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(key_features):
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(x=df[col], y=df['Price'], alpha=0.5)
        plt.title(f'Price vs. {col}', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/plots/price_vs_features.png')
    print("✅ Price vs. key features plots saved to 'static/plots/price_vs_features.png'")
    plt.close()







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
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Print Results
results_df = pd.DataFrame(results)
print("\n--- Evaluation Results ---")
print(results_df.sort_values(by='R-squared', ascending=False))

# Save evaluation results to a JSON file for the app's dashboard
dashboard_data = {
    'best_model_name': best_model_name,
    'best_r2': best_r2,
    'evaluation_results': results_df.to_dict(orient='records')
}
joblib.dump(dashboard_data, 'evaluation_results.json')
print("✅ Dashboard data saved to 'evaluation_results.json'")


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

# --- Generate and Save Plots for the Best Model ---
print("\n--- Generating plots for the best model ---")
best_model_pred = best_model.predict(X_test)

# 1. Actual vs Predicted Honey Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, best_model_pred, color='dodgerblue', edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Prices ({best_model_name})')
plt.grid(True)
plt.tight_layout()
plt.savefig('static/plots/actual_vs_predicted.png')
print("✅ Actual vs. Predicted plot saved to 'static/plots/actual_vs_predicted.png'")
plt.close() # Close the plot window

# 2. Distribution of Residuals
residuals = y_test - best_model_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='coral')
plt.title(f'Distribution of Residuals ({best_model_name})')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('static/plots/residuals_distribution.png')
print("✅ Residuals distribution plot saved to 'static/plots/residuals_distribution.png'")
plt.close() # Close the plot window

# 3. Feature Importance Plot (if applicable)
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Generating feature importance plot ---")
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(15) # Top 15 features
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png')
    print("✅ Feature importance plot saved to 'static/plots/feature_importance.png'")
    plt.close() # Close the plot window
