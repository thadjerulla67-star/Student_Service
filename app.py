import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from modules.data_processing import REQUIRED_COLUMNS, prepare_dataframe, validate_schema, build_model_dataset
from modules.exploratory_analysis import render_eda
from modules.prediction import FEATURES, predict_one, get_metrics
from modules.dashboard import render_dashboard

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.joblib"

st.set_page_config(page_title="Student Service SLA Monitoring", page_icon="📊", layout="wide")

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource
def load_saved_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def get_active_data():
    return st.session_state.get("df")

def train_and_save_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    model_df = build_model_dataset(df)
    if model_df.empty:
        raise ValueError("No eligible completed records with Processing_Days and SLA_Days were found.")

    X = model_df[FEATURES]
    y = model_df["SLA_Breached"]

    if y.nunique() < 2:
        raise ValueError("The eligible records contain only one SLA class; a two-class classifier cannot be trained.")

    categorical_features = [
        "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
        "Assigned_Office", "Required_Documents_Complete",
        "Submission_Weekday", "Queue_Category"
    ]
    numerical_features = ["Queue_Length", "SLA_Days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("numeric", "passthrough", numerical_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "model_type": "DecisionTreeClassifier",
        "random_state": 42,
        "source": "Adapted directly from the submitted midterm code"
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle, X_test, y_test

# Sidebar
st.sidebar.title("Student Service SLA System")
menu = st.sidebar.selectbox(
    "Module",
    ["Home", "Dataset", "EDA", "Prediction", "Dashboard", "System Design"]
)

st.sidebar.caption("Decision support only — final service decisions remain with authorized staff.")

if menu == "Home":
    st.title("Student Service Request Monitoring")
    st.subheader("Streamlit deployment of the Midterm Data-Mining Output")
    st.write(
        "This application reuses the midterm target, feature engineering, Decision Tree pipeline, "
        "and evaluation approach for predicting SLA breaches in completed student-service requests."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Target", "SLA_Breached")
    c2.metric("Model", "Decision Tree")
    c3.metric("Decision role", "Human review")

    st.info(
        "Upload student_service_requests.csv on the Dataset page. Because the dataset and fitted "
        "joblib model were not included with the supplied midterm code, this project can create and "
        "save the fitted pipeline from the uploaded dataset using the exact midterm procedure."
    )

    st.markdown("### Implemented modules")
    st.markdown("""
    - Dataset: CSV upload, schema validation, preview, nulls, duplicates
    - EDA: descriptive statistics, class distribution, and interactive charts
    - Prediction: one-request inference using the midterm feature set
    - Dashboard: workload/SLA KPIs, risk queue, and human follow-up guidance
    - System Design: requirements, actors, use case, and analytical workflow
    """)

elif menu == "Dataset":
    st.title("1. Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student_service_requests.csv", type=["csv"])

    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            errors = validate_schema(df)
            if errors:
                st.error("Schema validation failed.")
                for e in errors:
                    st.write(f"- {e}")
            elif df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                df = prepare_dataframe(df)
                st.session_state["df"] = df
                st.success(f"Validated dataset: {len(df):,} rows × {df.shape[1]} columns.")

                st.dataframe(df.head(20), use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", len(df))
                c2.metric("Columns", df.shape[1])
                c3.metric("Duplicate Request IDs", int(df["Request_ID"].duplicated().sum()))
                c4.metric("Invalid dates", int(df["Submission_Date"].isna().sum()))

                st.markdown("### Null-value summary")
                st.dataframe(df.isnull().sum().rename("Null_Count").to_frame(), use_container_width=True)

                st.markdown("### Data types")
                st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)

                st.markdown("### Model preparation")
                st.write("The saved pipeline is preferred. If it is not present, use the button below once to fit and save it.")
                if st.button("Prepare and save trained pipeline", type="primary"):
                    try:
                        bundle, X_test, y_test = train_and_save_model(df)
                        st.session_state["model_bundle"] = bundle
                        st.session_state["test_data"] = (X_test, y_test)
                        st.success("trained_model.joblib was created in models/.")
                    except Exception as exc:
                        st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
    else:
        st.warning("Upload the midterm CSV to begin.")

elif menu == "EDA":
    st.title("2. Exploratory Data Analysis")
    df = get_active_data()
    if df is None:
        st.warning("Upload a valid dataset on the Dataset page first.")
    else:
        render_eda(df)

elif menu == "Prediction":
    st.title("3. Predictive Modeling")
    df = get_active_data()
    bundle = st.session_state.get("model_bundle") or load_saved_model()

    if df is None:
        st.warning("Upload a valid dataset on the Dataset page first.")
    elif bundle is None:
        st.warning("No saved model is available. On Dataset, click “Prepare and save trained pipeline” once.")
    else:
        model = bundle["model"]
        st.caption(f"Model: {bundle.get('model_type', 'saved pipeline')} | Features: {len(bundle['feature_names'])}")

        options = {}
        for col in FEATURES:
            if col in ["Queue_Length", "SLA_Days"]:
                continue
            vals = df[col].dropna().astype(str).unique().tolist()
            options[col] = vals

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                service = st.selectbox("Service_Type", options["Service_Type"])
                requester = st.selectbox("Requester_Type", options["Requester_Type"])
                channel = st.selectbox("Requested_Channel", options["Requested_Channel"])
                priority = st.selectbox("Priority", options["Priority"])
                office = st.selectbox("Assigned_Office", options["Assigned_Office"])
            with c2:
                docs = st.selectbox("Required_Documents_Complete", options["Required_Documents_Complete"])
                weekday = st.selectbox("Submission_Weekday", options["Submission_Weekday"])
                queue_cat = st.selectbox("Queue_Category", options["Queue_Category"])
                queue_len = st.number_input("Queue_Length", min_value=0.0, value=5.0, step=1.0)
                sla_days = st.number_input("SLA_Days", min_value=0.0, value=3.0, step=1.0)

            submitted = st.form_submit_button("Predict SLA breach", type="primary")

        if submitted:
            row = {
                "Service_Type": service,
                "Requester_Type": requester,
                "Requested_Channel": channel,
                "Priority": priority,
                "Assigned_Office": office,
                "Queue_Length": queue_len,
                "Required_Documents_Complete": docs,
                "SLA_Days": sla_days,
                "Submission_Weekday": weekday,
                "Queue_Category": queue_cat,
            }
            pred, prob = predict_one(model, row)
            if pred == 1:
                st.error("Prediction: SLA BREACH RISK")
                st.warning("Human follow-up is recommended before the service request is finalized.")
            else:
                st.success("Prediction: NO SLA BREACH RISK")
            if prob is not None:
                st.write(f"Predicted breach probability: **{prob:.2%}**")
            st.caption("This prediction is decision support, not an automatic authorization or denial.")

        st.markdown("### Model metrics")
        test_data = st.session_state.get("test_data")
        if test_data:
            X_test, y_test = test_data
            metrics = get_metrics(model, X_test, y_test)
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        else:
            st.info("Metrics from the saved bundle are shown after the pipeline is prepared in this session.")

elif menu == "Dashboard":
    st.title("4. Decision-Support Dashboard")
    df = get_active_data()
    if df is None:
        st.warning("Upload a valid dataset on the Dataset page first.")
    else:
        render_dashboard(df)

elif menu == "System Design":
    st.title("5. Systems Analysis and Design Integration")

    st.markdown("### Actors")
    st.markdown("""
    1. **Student Services Staff** — review requests, inspect risk indicators, and record the final action.
    2. **System Administrator** — maintain the application, model artifact, and authorized access.
    3. **Decision-Support System** — validates data, computes analytics, and produces predictions.
    """)

    st.markdown("### Five evidence-based functional requirements")
    requirements = [
        "FR1 — The system shall accept a CSV student-service request dataset and validate required columns.",
        "FR2 — The system shall summarize nulls, duplicate Request_ID values, dates, and descriptive statistics.",
        "FR3 — The system shall visualize request volume, queue workload, and SLA-breach patterns.",
        "FR4 — The system shall apply the saved midterm preprocessing and Decision Tree pipeline to new request inputs.",
        "FR5 — The system shall present risk indicators and recommendations for authorized human review without automatically denying services.",
    ]
    for r in requirements:
        st.write("• " + r)

    st.markdown("### Non-functional requirements")
    nfr = [
        "NFR1 — Usability: modules are accessible through a clear sidebar navigation.",
        "NFR2 — Reliability: invalid files and missing columns produce explicit validation messages.",
        "NFR3 — Maintainability: data processing, EDA, prediction, and dashboard logic are separated into modules.",
    ]
    for r in nfr:
        st.write("• " + r)

    st.markdown("### Analytical decision workflow")
    st.code(
        "CSV upload → Schema validation → Cleaning/feature engineering → "
        "EDA/KPI review → Saved model inference → Risk/review queue → "
        "Authorized staff review → Final human decision"
    )

    diagram = BASE_DIR / "assets" / "use_case_diagram.png"
    if diagram.exists():
        st.image(str(diagram), caption="Updated use-case view for the implemented system.", use_container_width=True)

    st.markdown("### Model governance")
    st.write(
        "The model uses the same midterm target and feature list. Processing_Days and Status are not "
        "used as inference inputs because they are unavailable or outcome-related at the prediction checkpoint. "
        "The application exposes the prediction as decision support and keeps final authorization with staff."
    )
