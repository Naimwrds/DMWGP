import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import xgboost as xgb
import os
import numpy as np
from xgboost import XGBClassifier

image = Image.open('LoanApplication.jpg')
st.image(image, width=400)

logo = Image.open("Loan icon 2.png")
st.sidebar.image(logo, width=200)

st.sidebar.markdown("## About the App")
st.sidebar.write("""
This app provides insights into past loan applications and predicts loan statuses 
based on user inputs. Please navigate through the dashboard and prediction tools 
to explore more.
""")

# Print XGBoost version for debugging
st.write(f"XGBoost version: {xgb.__version__}")

# File paths
model_path = "best_xgboost_model.pkl"
scaler_path = "scaler.pkl"

# Define encoder paths explicitly
encoder_paths = {
    "person_gender": "person_gender_encoder.pkl",
    "person_education": "person_education_encoder.pkl",
    "person_home_ownership": "person_home_ownership_encoder.pkl",
    "loan_intent": "loan_intent_encoder.pkl",
    "previous_loan_defaults_on_file": "previous_loan_defaults_on_file_encoder.pkl",
}

# Define categorical columns
categorical_columns = list(encoder_paths.keys())

def load_file(file_path):
    """Safely load files with joblib."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

@st.cache_resource
def load_model(path):
    """Load the trained model."""
    try:
        model = load_file(path)
        if model is None:
            raise ValueError(f"Failed to load model from {path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scaler(path):
    """Load the scaler."""
    return load_file(path)

@st.cache_resource
def load_encoders(paths):
    """Load individual encoders explicitly."""
    encoders = {}
    for column, path in paths.items():
        try:
            if os.path.exists(path):
                encoders[column] = load_file(path)
                st.write(f"Loaded encoder for column: {column}")
            else:
                st.error(f"Encoder file not found: {path}")
        except Exception as e:
            st.error(f"Error loading encoder for {column}: {e}")
    return encoders

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    st.write(f"Model file found: {model_path}")


if "resources_loaded" not in st.session_state:
    try:
        st.session_state["model"] = load_model(model_path)
        if st.session_state["model"] is None:
            raise ValueError("Model is None. Check the model file and loading process.")

        st.session_state["scaler"] = load_scaler(scaler_path)

        if "encoders" not in st.session_state:
            st.session_state["encoders"] = load_encoders(encoder_paths)

        st.session_state["resources_loaded"] = True
    except Exception as e:
        st.error(f"Error initializing resources: {e}")
        st.session_state["resources_loaded"] = False

if st.session_state["model"] is not None:
    st.write("Model loaded successfully.")
else:
    st.error("Failed to load model.")


def encode_with_encoders(data, encoders, categorical_columns):
    """Apply encoders to specified categorical columns."""
    for column in categorical_columns:
        if column in encoders:
            try:
                encoder = encoders[column]
                data[column] = encoder.transform(data[column].astype(str))
                st.write(f"Encoded column: {column}")
            except Exception as e:
                st.error(f"Error encoding column '{column}': {e}")
                raise
        else:
            st.error(f"No encoder found for column '{column}'.")
            raise ValueError(f"Missing encoder for {column}")
    return data

def show_dashboard():
    st.title("Loan Application Dashboard")

    try:
        # Load dataset
        dataset = pd.read_csv("loan_data.csv")

        # 1. Calculate accepted and rejected percentages
        accepted_count = dataset[dataset["loan_status"] == 1].shape[0]
        rejected_count = dataset[dataset["loan_status"] == 0].shape[0]
        total_count = dataset.shape[0]

        accepted_percentage = (accepted_count / total_count) * 100
        rejected_percentage = (rejected_count / total_count) * 100

        # 2. Most common loan intent categories
        loan_intent_counts = dataset["loan_intent"].value_counts()

        # 3. Display average,min and max interest rate so far
        avg_loan_interest_rate = dataset["loan_int_rate"].mean()
        min_loan_interest_rate = dataset["loan_int_rate"].min()
        max_loan_interest_rate = dataset["loan_int_rate"].max()

        # 4. Most education level applied for loan
        loan_person_education_counts = dataset["person_education"].value_counts()
        
        # 5. Average, min, and max loan amount
        avg_loan_amount = dataset["loan_amnt"].mean()
        min_loan_amount = dataset["loan_amnt"].min()
        max_loan_amount = dataset["loan_amnt"].max()

        # 6. Income distribution for loan applicants
        income_stats = dataset["person_income"].describe()

        # 7. Loan acceptance rates based on credit score
        accepted_loans = dataset[dataset["loan_status"] == 1]
        rejected_loans = dataset[dataset["loan_status"] == 0]

        min_credit_score_accepted = accepted_loans["credit_score"].min()
        max_credit_score_rejected = rejected_loans["credit_score"].max()
        
        # 8. Percentage of loans with previous defaults
        previous_default_yes = dataset[dataset["previous_loan_defaults_on_file"] == "Yes"].shape[0]
        previous_default_no = dataset[dataset["previous_loan_defaults_on_file"] == "No"].shape[0]
        total_loans = dataset.shape[0]

        percentage_defaults = (previous_default_yes / total_loans) * 100
        percentage_nondefaults = (previous_default_no / total_loans) * 100
        
        # 1. Loan accepted & rejected
        st.write("### 1. Loan Application Summary")
        st.markdown(f"<h4 style='color:green;'>Accepted Applications: {accepted_count} ({accepted_percentage:.2f}%)</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:red;'>Rejected Applications: {rejected_count} ({rejected_percentage:.2f}%)</h4>", unsafe_allow_html=True)
        st.bar_chart({"Accepted": accepted_count, "Rejected": rejected_count})
            
        # 2. Loan Intent Distribution
        st.write("### 2. Most Common Loan Intents")
        st.bar_chart(loan_intent_counts)
        
        # 3. Display average, min, and max interest rate
        st.write("### 3. Interest Rate")
        st.markdown(f"<h4 style='color:blue;'>Average Loan Interest Rate: {avg_loan_interest_rate:.2f}%</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:green;'>Minimum Loan Interest Rate: {min_loan_interest_rate:.2f}%</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:red;'>Maximum Loan Interest Rate: {max_loan_interest_rate:.2f}%</h4>", unsafe_allow_html=True)     
        
        # 4. Most Education Level applied for Loan
        st.write("### 4. Most Education Level applied for Loan")
        st.bar_chart(loan_person_education_counts)

        # 5. Display average, min, and max loan amount
        st.write("### 5. Loan Amount")
        st.markdown(f"<h4 style='color:blue;'>Average Loan Amount: {avg_loan_amount:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:green;'>Minimum Loan Amount: {min_loan_amount:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:red;'>Maximum Loan Amount: {max_loan_amount:.2f}</h4>", unsafe_allow_html=True)
        
        # 6. Income Analysis
        st.write("### 6. Income Distribution Statistics")
        st.write(income_stats)
        
        # 7. Credit Score Analysis
        st.write("### 7. Credit Score Analysis")
        st.markdown(f"<h4 style='color:green;'>Minimum Credit Score for Accepted Loans: {min_credit_score_accepted:.2f}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:red;'>Maximum Credit Score for Rejected Loans: {max_credit_score_rejected:.2f}</h4>", unsafe_allow_html=True)   
        
        # 8. Previous Defaults
        st.write("### 8. Previous Defaults Analysis")
        # Loans with Previous Defaults
        st.metric(
        label="Loans with Previous Defaults",
        value=f"{previous_default_yes}",
        delta=f"{percentage_defaults:.2f}%",
        delta_color="normal"  
        )
        # Loans without Previous Defaults
        st.metric(
            label="Loans without Previous Defaults",
            value=f"{previous_default_no}",
            delta=f"{percentage_nondefaults:.2f}%",
            delta_color="inverse"  
        )

        
    except Exception as e:
        st.error(f"Error displaying dashboard: {e}")

def map_with_dict(value, mapping):
    """Map a value using a dictionary."""
    return mapping.get(value, -1)  # Return -1 for unknown values


def encode_with_encoders(data, encoders, categorical_columns):
    """Apply encoders (or mappings) to specified categorical columns."""
    for column in categorical_columns:
        if column in encoders:
            encoder = encoders[column]
            try:
                # Check if encoder is a dictionary or an object
                if isinstance(encoder, dict):
                    data[column] = data[column].apply(lambda x: map_with_dict(x, encoder))
                else:
                    data[column] = encoder.transform(data[column].astype(str))
                st.write(f"Encoded column: {column}")  # Debug: Confirm successful encoding
            except Exception as e:
                st.error(f"Error encoding column '{column}': {e}")
                raise
        else:
            st.error(f"No encoder found for column '{column}'.")
            raise ValueError(f"Missing encoder for {column}")
    return data


def show_prediction():
    st.title("Loan Status Prediction")

    if not st.session_state.get("resources_loaded", False):
        st.error("Model and scaler are not loaded. Please check the resources.")
        return

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    encoders = st.session_state["encoders"]

    # User inputs
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master"])
    person_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50, value=2)
    person_home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT"])
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, value=0.3)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

    # Prepare input with original feature names
    input_data = pd.DataFrame([[
        person_age, person_gender, person_education, person_income,
        person_emp_exp, person_home_ownership, loan_amnt, loan_intent,
        loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
        credit_score, previous_loan_defaults_on_file
    ]], columns=[
        "person_age", "person_gender", "person_education", "person_income",
        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file"
    ])

    # Apply encoders to categorical columns
    try:
        input_data = encode_with_encoders(input_data, encoders, categorical_columns)
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        return

    # Ensure all features are numeric
    if not all(input_data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
        st.error("Not all features are numeric after encoding.")
        return

    # Scale the data
    try:
        scaled_data = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        return

    # Predict loan status
    if st.button("Predict Loan Status"):
        try:
            prediction = model.predict(scaled_data)
            status = "Accepted" if prediction[0] == 1 else "Rejected"
            st.write(f"### Loan Status: {status}")
        except Exception as e:
            st.error(f"Prediction error: {e}")



# Main function
def main():
    st.sidebar.title("Loan Prediction App")
    page = st.sidebar.selectbox("Select a Page", ["Dashboard", "Predict"])

    if page == "Dashboard":
        show_dashboard()
    elif page == "Predict":
        show_prediction()

if __name__ == "__main__":
    main()
