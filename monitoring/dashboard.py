
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

st.title("📊 Model Monitoring Dashboard")

LOG_FILE = "logs/predictions.jsonl"

# Load logs safely
data = []
try:
    with open(LOG_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
except FileNotFoundError:
    st.error("No logs found yet.")
    st.stop()

df = pd.DataFrame(data)

# ---- Latest predictions ----
st.subheader("Latest Predictions")
st.dataframe(df.tail(10))

# ---- Prediction distribution ----
st.subheader("Prediction Distribution")
st.bar_chart(df["prediction"].value_counts())

# ---- Probability distribution (FIXED) ----
st.subheader("Probability Distribution")

fig, ax = plt.subplots()
ax.hist(df["probability"], bins=10)

st.pyplot(fig)

# ---- Probability trend over time ----
st.subheader("Probability Trend")
st.line_chart(df["probability"])

# ---- Confidence stats ----
st.subheader("Confidence Stats")
st.write(df["probability"].describe())



