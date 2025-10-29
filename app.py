from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import joblib

# --- Initialize Flask App ---
app = Flask(__name__)

# Initialize artifacts to None or empty
pollen_types = []
dashboard_data = {}
# --- Load Artifacts at startup ---
try:
    model = joblib.load('best_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    pollen_types = joblib.load('pollen_types.pkl')
    dashboard_data = joblib.load('evaluation_results.json')
    print("✅ Model and artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading artifacts: {e}")
    print("Please run 'python main.py' to generate model artifacts.")
    model = None # Set to None to handle gracefully

@app.route('/')
def dashboard():
    """Renders the main dashboard page with model evaluation results."""
    if not dashboard_data:
        return "Dashboard data not loaded. Please run 'main.py' to generate artifacts.", 500
    # Sort results for display
    sorted_results = sorted(dashboard_data['evaluation_results'], key=lambda x: x['R-squared'], reverse=True)
    dashboard_data['evaluation_results'] = sorted_results
    return render_template('dashboard.html', dashboard_data=dashboard_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handle GET requests by showing the form.
    Handle POST requests by processing form data and returning a prediction.
    """
    if model is None:
        return "Model not loaded. Please check server logs and ensure artifacts exist.", 500
    last_input = {}

    prediction_text = ""
    if request.method == 'POST':
        # --- Get form data ---
        form_data = request.form.to_dict()
        
        # Convert numeric fields from string to float
        for key, value in form_data.items():
            if key != 'Pollen_analysis':
                try:
                    form_data[key] = float(value)
                except ValueError:
                    # Handle cases where a field might be empty or non-numeric
                    return "Invalid input for numeric fields.", 400
        
        # Store form data to repopulate the form
        last_input = form_data.copy()

        # --- Create DataFrame for prediction ---
        input_df = pd.DataFrame([form_data])

        # --- Preprocess the input to match the model's training data ---
        processed_input = pd.get_dummies(input_df, columns=['Pollen_analysis'])

        # Align columns with the model's training columns, filling missing with 0
        final_input = processed_input.reindex(columns=model_columns, fill_value=0)

        # --- Make prediction ---
        prediction = model.predict(final_input)
        
        # Format the prediction text to be displayed
        prediction_text = f"Predicted Price: ${prediction[0]:.2f}"

    # On GET request or after POST, render the main page
    return render_template(
        'index.html', 
        pollen_types=pollen_types, 
        prediction_text=prediction_text,
        last_input=last_input
    )

@app.route('/plots')
def plots():
    """Renders the page that displays model plots, checking for each plot's existence."""
    plot_files = {
        'correlation': 'static/plots/correlation_matrix.png',
        'actual_vs_predicted': 'static/plots/actual_vs_predicted.png',
        'residuals': 'static/plots/residuals_distribution.png',
        'feature_importance': 'static/plots/feature_importance.png'
    }
    
    # Create a dictionary telling the template which plots exist
    plots_exist = {name: os.path.exists(path) for name, path in plot_files.items()}
    return render_template('plots.html', plots_exist=plots_exist)

@app.route('/analysis')
def analysis():
    """Renders the page with EDA plots of the dataset."""
    analysis_plot_files = {
        'distributions': 'static/plots/feature_distributions.png',
        'price_vs_features': 'static/plots/price_vs_features.png',
        'correlation': 'static/plots/correlation_matrix.png' # Also relevant here
    }
    
    # Create a dictionary telling the template which plots exist
    plots_exist = {name: os.path.exists(path) for name, path in analysis_plot_files.items()}
    return render_template('analysis.html', plots_exist=plots_exist)


if __name__ == '__main__':
    # To run the app:
    # 1. Make sure you have Flask installed: pip install Flask
    # 2. Create a 'templates' folder with 'index.html' inside.
    # 3. Create a 'static' folder with 'style.css' inside.
    # 4. Run this script: python app.py
    # 5. Open your browser to http://127.0.0.1:5000
    app.run(debug=True)
