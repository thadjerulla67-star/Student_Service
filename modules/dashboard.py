import pandas as pd
import plotly.express as px
import streamlit as st

def render_dashboard(df):
    completed = df[
        (df["Status"] == "Completed") &
        df["Processing_Days"].notna() &
        df["SLA_Days"].notna()
    ].copy()

    if completed.empty:
        st.warning("No completed records with processing and SLA days are available for the dashboard.")
        return

    breach_rate = completed["SLA_Breached"].mean() * 100
    avg_queue = df["Queue_Length"].mean()
    avg_processing = completed["Processing_Days"].mean()
    reopened = (df["Reopened"].astype(str).str.lower() == "yes").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total requests", f"{len(df):,}")
    c2.metric("Completed requests", f"{len(completed):,}")
    c3.metric("SLA breach rate", f"{breach_rate:.1f}%")
    c4.metric("Average queue length", f"{avg_queue:.1f}")

    st.metric("Average processing days (completed)", f"{avg_processing:.2f}")
    st.metric("Reopened rate", f"{reopened:.1f}%")

    st.markdown("### Risk / review queue")
    risk = completed.copy()
    risk["Delay_Over_SLA"] = risk["Processing_Days"] - risk["SLA_Days"]
    risk = risk.sort_values(["SLA_Breached", "Delay_Over_SLA", "Queue_Length"], ascending=False)
    cols = [c for c in [
        "Request_ID", "Service_Type", "Priority", "Assigned_Office",
        "Queue_Length", "Processing_Days", "SLA_Days", "Delay_Over_SLA",
        "SLA_Breached", "Reopened"
    ] if c in risk.columns]
    st.dataframe(risk[cols].head(25), use_container_width=True)

    fig = px.scatter(
        completed, x="Queue_Length", y="Processing_Days",
        color="SLA_Breached", hover_data=["Service_Type", "Priority", "Assigned_Office"],
        title="Queue Length vs Processing Days"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recommended human follow-up")
    st.write(
        "Prioritize requests with a breach indicator, long queues, or processing time above the SLA. "
        "Authorized staff should verify the underlying request details before recording any final action."
    )
