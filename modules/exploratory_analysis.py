import pandas as pd
import plotly.express as px
import streamlit as st

def render_eda(df):
    completed = df[
        (df["Status"] == "Completed") &
        df["Processing_Days"].notna() &
        df["SLA_Days"].notna()
    ].copy()

    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("### Class distribution")
    class_counts = completed["SLA_Breached"].value_counts().rename(index={0: "No Breach", 1: "Breach"})
    st.dataframe(class_counts.rename("Count").to_frame(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        service_counts = df["Service_Type"].value_counts().reset_index()
        service_counts.columns = ["Service_Type", "Requests"]
        fig = px.bar(service_counts, x="Requests", y="Service_Type", orientation="h",
                     title="Student Service Requests by Service Type")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        queue = completed.groupby("Queue_Category", observed=False)["SLA_Breached"].mean().mul(100).reset_index()
        queue.columns = ["Queue_Category", "Breach_Rate"]
        fig = px.bar(queue, x="Queue_Category", y="Breach_Rate",
                     title="SLA Breach Rate by Queue Category",
                     labels={"Breach_Rate": "SLA Breach Rate (%)"})
        st.plotly_chart(fig, use_container_width=True)

    office = completed.groupby("Assigned_Office")["SLA_Breached"].mean().mul(100).reset_index()
    fig = px.bar(office.sort_values("SLA_Breached", ascending=False),
                 x="Assigned_Office", y="SLA_Breached",
                 title="SLA Breach Rate by Assigned Office",
                 labels={"SLA_Breached": "SLA Breach Rate (%)"})
    st.plotly_chart(fig, use_container_width=True)

    priority = completed.groupby("Priority")["SLA_Breached"].mean().mul(100).reset_index()
    fig = px.bar(priority, x="Priority", y="SLA_Breached",
                 title="SLA Breach Rate by Priority",
                 labels={"SLA_Breached": "SLA Breach Rate (%)"})
    st.plotly_chart(fig, use_container_width=True)
