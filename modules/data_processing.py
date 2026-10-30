import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Request_ID", "Submission_Date", "Service_Type", "Requester_Type",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Processing_Days", "SLA_Days", "Required_Documents_Complete",
    "Status", "Satisfaction_Score", "Reopened"
]

def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    errors = []
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
    if "Request_ID" in df.columns and df["Request_ID"].duplicated().any():
        errors.append(f"Duplicate Request_ID values: {int(df['Request_ID'].duplicated().sum())}")
    return errors

def prepare_dataframe(df):
    df = df.copy()
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    df["Queue_Category"] = pd.cut(
        df["Queue_Length"],
        bins=[-1, 5, 15, float("inf")],
        labels=["Low", "Medium", "High"]
    )
    df["SLA_Breached"] = np.where(
        df["Processing_Days"].notna() & df["SLA_Days"].notna() &
        (df["Processing_Days"] > df["SLA_Days"]),
        1, 0
    )
    return df

def build_model_dataset(df):
    work = prepare_dataframe(df)
    return work[
        (work["Status"] == "Completed") &
        work["Processing_Days"].notna() &
        work["SLA_Days"].notna()
    ].copy()
