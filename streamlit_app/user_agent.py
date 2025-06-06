import os, sys, ast
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq

# allow "import agents" no matter where we run Streamlit from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.phi_forecaster import run_forecast, _robust_parse_dates, _clean_numeric

# ---------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="📈 Universal Sales Forecast Assistant")
st.title("📈 Universal Sales Forecast Assistant")

uploaded = st.file_uploader("Upload sales CSV", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.write("📄 Preview", raw_df.head())

    # --- LLM agent finds date/sales cols -----------------------------------
    agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Identify date and sales columns and whether dataset is transaction-level.",
            "Respond ONLY as Python dict: {'date': '...', 'sales': '...', 'is_transaction_level': true}",
        ],
    )
    mapping = ast.literal_eval(
        agent.run(
            f"Dataframe columns: {raw_df.columns.tolist()}. Return dict only."
        ).content
    )
    date_col = mapping["date"]
    sales_col = mapping["sales"]
    tx_level = mapping.get("is_transaction_level", False)

    horizon = st.slider("Days to forecast", 7, 90, 30)

    if st.button("Run forecast"):
        df = raw_df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"})
        df["ds"] = _robust_parse_dates(df["ds"])
        df["y"] = _clean_numeric(df["y"])
        df = df.dropna(subset=["ds", "y"])
        df = df[df["y"] > 0]

        if tx_level:
            df["ds"] = df["ds"].dt.date
            df = df.groupby("ds", as_index=False)["y"].sum()
            df["ds"] = pd.to_datetime(df["ds"])

        st.info(f"Training window: {df['ds'].min().date()} → {df['ds'].max().date()}")

        fc = run_forecast(df.to_dict(orient="records"), future_days=horizon)
        fc_df = pd.DataFrame(fc)

        st.success("✅ Forecast complete")

        # ---------- plot ---------------
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fc_df["ds"],
                y=fc_df["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="magenta"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["ds"],
                y=fc_df["yhat_upper"],
                mode="lines",
                showlegend=False,
                line=dict(width=0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["ds"],
                y=fc_df["yhat_lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(200,100,250,.25)",
                line=dict(width=0),
                name="Confidence",
            )
        )
        fig.update_layout(
            title="🧠 Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("🔮 Forecast table", fc_df[["ds", "yhat"]])
