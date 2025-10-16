# This is magic line command
# This command is for Notebooks. This command make this cell a python file

import streamlit as st
import pandas as pd
import joblib

# --- Caching Functions to Load Files ---
@st.cache_data
def load_model():
    """Loads the saved machine learning model."""
    model = joblib.load('best_model.pkl')
    return model

@st.cache_data
def load_columns():
    """Loads the list of model columns."""
    model_cols = joblib.load('model_columns.pkl')
    return model_cols

@st.cache_data
def load_pollen_types():
    """Loads the unique pollen types for the dropdown."""
    types = joblib.load('pollen_types.pkl')
    return types

# --- Load Artifacts ---
model = load_model()
model_columns = load_columns()
pollen_types = load_pollen_types()

# --- Page Configuration ---
st.set_page_config(
    page_title="Honey Price Predictor",
    page_icon="🍯",
    layout="centered"
)

# --- UI Elements ---
st.title("🍯 Honey Price Predictor")
st.write("""
Enter the characteristics of the honey to get a price prediction.
You can directly type the values or use the `+` and `-` buttons.
""")

# --- Input Form in Sidebar ---
st.sidebar.header("Input Honey Features:")

def user_input_features():
    """Creates sidebar elements and returns user inputs as a DataFrame."""
    pollen_analysis = st.sidebar.selectbox('Pollen Analysis (Type of Honey)', pollen_types)

    # --- All number_input fields now have step=0.01 and format="%.2f" ---
    cs = st.sidebar.number_input('Color (CS)', min_value=1.0, max_value=10.0, value=5.50, step=0.01, format="%.2f")
    density = st.sidebar.number_input('Density (g/cm³)', min_value=1.21, max_value=1.86, value=1.54, step=0.01, format="%.2f")
    wc = st.sidebar.number_input('Water Content (WC %)', min_value=12.0, max_value=25.0, value=18.50, step=0.01, format="%.2f")
    ph = st.sidebar.number_input('pH', min_value=2.50, max_value=7.50, value=5.00, step=0.01, format="%.2f")
    ec = st.sidebar.number_input('Electrical Conductivity (EC)', min_value=0.70, max_value=0.90, value=0.80, step=0.01, format="%.2f")
    f = st.sidebar.number_input('Fructose (F %)', min_value=20.0, max_value=50.0, value=34.97, step=0.01, format="%.2f")
    g = st.sidebar.number_input('Glucose (G %)', min_value=20.0, max_value=45.0, value=32.50, step=0.01, format="%.2f")
    viscosity = st.sidebar.number_input('Viscosity (mPa·s)', min_value=1500.0, max_value=10000.0, value=5753.0, step=0.01, format="%.2f")
    purity = st.sidebar.number_input('Purity (0.0 to 1.0 scale)', min_value=0.61, max_value=1.0, value=0.82, step=0.01, format="%.2f")

    data = {
        'CS': cs, 'Density': density, 'WC': wc, 'pH': ph, 'EC': ec, 'F': f, 'G': g,
        'Viscosity': viscosity, 'Purity': purity, 'Pollen_analysis': pollen_analysis
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user's selected features in the main area
st.subheader('Your Input Features')
st.dataframe(input_df)

# --- Prediction Logic ---
if st.sidebar.button('Predict Price'):
    # Preprocess the input to match the model's training data
    processed_input = pd.get_dummies(input_df, columns=['Pollen_analysis'])

    # Align columns with the model's training columns, filling missing with 0
    final_input = processed_input.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(final_input)

    # Display prediction
    st.subheader('Predicted Price')
    st.success(f"**${prediction[0]:.2f}**")
else:
    st.sidebar.info("Adjust the values and click 'Predict Price' to see the result.")
