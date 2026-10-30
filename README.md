# Student Service SLA Monitoring — Streamlit

This project converts the supplied midterm notebook code into a Streamlit decision-support application.

## Midterm continuity

The supplied midterm code:
- loads `student_service_requests.csv`;
- creates `Submission_Weekday` and `Queue_Category`;
- defines `SLA_Breached = Processing_Days > SLA_Days`;
- keeps completed records with non-null processing/SLA days;
- uses the listed categorical and numerical features;
- uses a `DecisionTreeClassifier(random_state=42)`;
- evaluates accuracy, precision, recall, F1, and a majority baseline.

The Streamlit application preserves that target and feature set.

## Important input

The uploaded materials contained the midterm code and the Streamlit laboratory instructions, but not the actual CSV dataset or a pre-fitted `trained_model.joblib`. Therefore the application includes a one-time **Prepare and save trained pipeline** action that follows the supplied midterm training procedure and writes `models/trained_model.joblib`. After that, the app loads the saved pipeline with `st.cache_resource` rather than retraining on each page refresh.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `student_service_requests.csv` from the Dataset page.

## Deployment

1. Put this project in a GitHub repository.
2. Include the dataset only if it is synthetic or properly authorized.
3. Deploy `app.py` through Streamlit Community Cloud.
4. Test CSV upload, invalid schema, prediction inputs, charts, dashboard, and System Design.
5. Record screenshots and the deployed URL for submission.

## Files

- `app.py`
- `requirements.txt`
- `modules/data_processing.py`
- `modules/exploratory_analysis.py`
- `modules/prediction.py`
- `modules/dashboard.py`
- `models/trained_model.joblib` — generated after model preparation
- `assets/use_case_diagram.png`
