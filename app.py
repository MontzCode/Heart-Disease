import streamlit as st
import pandas as pd
import joblib
import os
import logging # Optional: for logging within the app if needed

# --- PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
# ------------------------------------------------------

# --- Configuration ---
# Assuming app.py is in the project root, adjust path if needed
MODEL_DIR = "models"
PIPELINE_NAME = "logistic_regression_pipeline.joblib" # Make sure this matches the saved file
PIPELINE_PATH = os.path.join(MODEL_DIR, PIPELINE_NAME)

# Define the expected order of features based on training data
# IMPORTANT: This order MUST match the order of columns in X_train used for fitting the pipeline
EXPECTED_FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# --- Logging Setup (Optional) ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# --- Load Model Pipeline ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model_pipeline(path): # Function definition (no indent)
    """Loads the saved Scikit-learn pipeline."""
    # Code inside the function is indented
    try:
        pipeline = joblib.load(path)
        # logger.info("Model pipeline loaded successfully.")
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model pipeline file not found at {path}. Make sure the model has been trained and saved.")
        # logger.error(f"Pipeline file not found: {path}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        # logger.error(f"Error loading pipeline: {e}", exc_info=True)
        return None
# End of function definition

# Call the function - this is top-level code (no indent)
pipeline = load_model_pipeline(PIPELINE_PATH)

# --- UI Mappings ---
# Create user-friendly mappings for categorical features
# Values MUST correspond to the numerical encoding used during training
sex_map = {"Female": 0, "Male": 1}
cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal Pain": 3, "Asymptomatic": 4}
fbs_map = {"False (< 120 mg/dl)": 0, "True (> 120 mg/dl)": 1}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Probable/definite left ventricular hypertrophy": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
thal_map = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

# --- Streamlit App Layout ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details to predict the likelihood of heart disease.")
st.markdown("---")

if pipeline is None:
    st.warning("Model could not be loaded. Please check the logs or ensure the model file exists.")
else:
    # Use columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        sex_label = st.selectbox("Sex", options=list(sex_map.keys()), index=1) # Default Male

    with col2:
        st.subheader("Symptoms & History")
        cp_label = st.selectbox("Chest Pain Type (cp)", options=list(cp_map.keys()), index=3) # Default Asymptomatic
        exang_label = st.selectbox("Exercise Induced Angina (exang)", options=list(exang_map.keys()), index=0) # Default No
        fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=list(fbs_map.keys()), index=0) # Default False

    with col3:
        st.subheader("Medical Measurements")
        trestbps = st.number_input("Resting Blood Pressure (trestbps) (mm Hg)", min_value=50, max_value=250, value=120, step=1)
        chol = st.number_input("Serum Cholesterol (chol) (mg/dl)", min_value=100, max_value=600, value=200, step=1)
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150, step=1)


    st.markdown("---")
    st.subheader("Exercise & ECG Results")
    col4, col5, col6 = st.columns(3)

    with col4:
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope_label = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=list(slope_map.keys()), index=1) # Default Flat

    with col5:
         restecg_label = st.selectbox("Resting Electrocardiographic Results (restecg)", options=list(restecg_map.keys()), index=0) # Default Normal
         ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3], index=0) # Default 0

    with col6:
        thal_label = st.selectbox("Thalium Stress Test Result (thal)", options=list(thal_map.keys()), index=0) # Default Normal


    st.markdown("---")

    # --- Prediction Logic ---
    if st.button("Predict Heart Disease", type="primary"):
        # 1. Collect and map inputs
        input_data = {
            'age': age,
            'sex': sex_map[sex_label],
            'cp': cp_map[cp_label],
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs_map[fbs_label],
            'restecg': restecg_map[restecg_label],
            'thalach': thalach,
            'exang': exang_map[exang_label],
            'oldpeak': oldpeak,
            'slope': slope_map[slope_label],
            'ca': ca,
            'thal': thal_map[thal_label]
        }

        # 2. Create DataFrame in the correct order
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[EXPECTED_FEATURE_ORDER] # Ensure column order

            # 3. Make Prediction
            prediction = pipeline.predict(input_df)[0]
            prediction_proba = pipeline.predict_proba(input_df)[0]

            # 4. Display Results
            st.subheader("Prediction Result")
            if prediction == 0:
                st.success("Prediction: **No Heart Disease** (Class 0)")
            else:
                st.error("Prediction: **Heart Disease Present** (Class 1)")

            # Display probability
            st.metric(
                label=f"Probability of Heart Disease (Class 1)",
                value=f"{prediction_proba[1]:.2%}" # Display probability of class 1
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            # logger.error(f"Prediction error: {e}", exc_info=True)

# --- Footer / Info ---
st.markdown("---")
st.markdown("Disclaimer: This prediction is based on a machine learning model trained on the UCI Heart Disease dataset and should not be used as a substitute for professional medical advice.")