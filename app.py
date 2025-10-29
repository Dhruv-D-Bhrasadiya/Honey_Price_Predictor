from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

# --- Initialize Flask App ---
app = Flask(__name__)

# Initialize artifacts to None or empty
pollen_types = []
# --- Load Artifacts at startup ---
try:
    model = joblib.load('best_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    pollen_types = joblib.load('pollen_types.pkl')
    print("✅ Model and artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading artifacts: {e}")
    print("Please run 'python main.py' to generate model artifacts.")
    model = None # Set to None to handle gracefully

@app.route('/', methods=['GET', 'POST'])
def predict():
    """
    Handle GET requests by showing the form.
    Handle POST requests by processing form data and returning a prediction.
    """
    if model is None:
        return "Model not loaded. Please check server logs and ensure artifacts exist.", 500

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
        prediction_text=prediction_text
    )

if __name__ == '__main__':
    # To run the app:
    # 1. Make sure you have Flask installed: pip install Flask
    # 2. Create a 'templates' folder with 'index.html' inside.
    # 3. Create a 'static' folder with 'style.css' inside.
    # 4. Run this script: python app.py
    # 5. Open your browser to http://127.0.0.1:5000
    app.run(debug=True)
