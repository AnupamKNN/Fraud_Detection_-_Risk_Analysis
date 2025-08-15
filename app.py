import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# LangChain + ChatGroq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard with AI Insights")

# -------------------------
# Load model/preprocessor
# -------------------------
@st.cache_resource
def load_artifacts():
    preprocessor_path = os.path.join("final_models", "preprocessor.pkl")
    model_path = os.path.join("final_models", "model.pkl")
    return joblib.load(preprocessor_path), joblib.load(model_path)

preprocessor, model = load_artifacts()

# -------------------------
# Feature engineering
# -------------------------
def add_engineered_features(df):
    start_date = pd.to_datetime('2020-01-01')
    df['TransactionDate'] = start_date + pd.to_timedelta(df['Time'], unit='s')
    df['Hour'] = df['TransactionDate'].dt.hour
    df['ElapsedDays'] = (df['TransactionDate'] - start_date).dt.days
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    # Readable time formats
    df['Readable_Time_12hr'] = df['TransactionDate'].dt.strftime('%I:%M:%S %p')
    df['Readable_Time_24hr'] = df['TransactionDate'].dt.strftime('%H:%M:%S')
    df.drop(columns=['Hour'], inplace=True, errors='ignore')
    return df

# Fraud label colours
def fraud_color(val):
    if val == 'Fraud':
        return 'background-color: #ffcccc; color: red; font-weight: bold;'
    elif val == 'Not Fraud':
        return 'background-color: #d4edda; color: green; font-weight: bold;'
    return ''

# -------------------------
# LLM and prompts
# -------------------------
def get_llm():
    if not GROQ_API_KEY:
        st.warning("🚨 Missing GROQ_API_KEY")
        return None
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.5)

explanation_prompt = PromptTemplate(
    input_variables=["transaction", "fraud_prob"],
    template=(
        "You are a fraud detection expert.\n"
        "A transaction has a fraud probability of {fraud_prob:.2f}.\n"
        "Transaction details: {transaction}\n"
        "Explain in plain language why this transaction might be flagged as fraud or considered safe."
    )
)

batch_summary_prompt = PromptTemplate(
    input_variables=["summary_text"],
    template=(
        "You are a fraud analytics expert. Based on the following statistics:\n"
        "{summary_text}\n"
        "Write a concise summary highlighting the key insights."
    )
)

def explain_with_ai(llm, transaction_row, fraud_prob):
    chain = LLMChain(llm=llm, prompt=explanation_prompt)
    return chain.run(transaction=transaction_row.to_dict(), fraud_prob=fraud_prob)

# -------------------------
# File upload and analysis
# -------------------------
uploaded_file = st.file_uploader("📂 Upload transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = add_engineered_features(df)

    # Threshold slider
    threshold = st.slider("🔧 Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

    # Run predictions
    X_processed = preprocessor.transform(df)
    fraud_probs = model.predict_proba(X_processed)[:, 1]
    df['Fraud_Probability'] = fraud_probs
    df['Fraud_Prediction'] = (fraud_probs >= threshold).astype(int)
    df['Fraud_Label'] = df['Fraud_Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

    # KPIs
    total_tx = len(df)
    n_fraud = df['Fraud_Prediction'].sum()
    pct_fraud = (n_fraud / total_tx) * 100
    avg_prob = df['Fraud_Probability'].mean()
    max_amount_fraud = df.loc[df['Fraud_Prediction'] == 1, 'Amount'].max() if n_fraud > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Transactions", total_tx)
    kpi2.metric("Frauds Detected", n_fraud)
    kpi3.metric("Fraud %", f"{pct_fraud:.2f}%")
    kpi4.metric("Max Fraud Amount", f"${max_amount_fraud:,.2f}")

    # Fraud probability distribution
    fig, ax = plt.subplots(figsize = (6, 3))
    ax.hist(df['Fraud_Probability'], bins=30, color='skyblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
    ax.set_title("Fraud Probability Distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # All predictions table
    st.subheader("📊 All Predictions")
    styled_df = df.style.applymap(fraud_color, subset=['Fraud_Label'])
    st.dataframe(styled_df, use_container_width=True)

    csv_all = io.StringIO()
    df.to_csv(csv_all, index=False)
    st.download_button("📥 Download All Predictions", csv_all.getvalue(), "fraud_predictions.csv", "text/csv")

    # Fraud-only table with click-to-explain
    st.subheader("🚨 Fraudulent Transactions Only")
    fraud_df = df[df['Fraud_Prediction'] == 1]

    if not fraud_df.empty:
        st.dataframe(fraud_df.style.applymap(fraud_color, subset=['Fraud_Label']), use_container_width=True)
        
        fraud_indices = fraud_df.index.tolist()
        selected_idx = st.selectbox("Select Fraudulent Transaction to Explain", fraud_indices)

        if st.button("Explain This Fraudulent Transaction"):
            llm = get_llm()
            if llm:
                with st.spinner("Generating explanation..."):
                    explanation = explain_with_ai(
                        llm,
                        fraud_df.loc[selected_idx].drop(labels=['Fraud_Probability', 'Fraud_Prediction', 'Fraud_Label']),
                        fraud_df.loc[selected_idx]['Fraud_Probability']
                    )
                st.markdown("**📝 Explanation:**")
                st.write(explanation)

        csv_fraud = io.StringIO()
        fraud_df.to_csv(csv_fraud, index=False)
        st.download_button("📥 Download Fraudulent Records", csv_fraud.getvalue(), "fraud_only.csv", "text/csv")
    else:
        st.info("No fraudulent transactions detected with current threshold.")

    # Batch AI Summary
    if st.button("📝 Generate AI Batch Summary"):
        llm = get_llm()
        if llm:
            stats_text = (
                f"Total transactions: {total_tx}, "
                f"Frauds detected: {n_fraud} ({pct_fraud:.2f}%), "
                f"Average fraud probability: {avg_prob:.2f}, "
                f"Max fraud amount: {max_amount_fraud}"
            )
            chain = LLMChain(llm=llm, prompt=batch_summary_prompt)
            with st.spinner("Generating AI summary..."):
                summary = chain.run(summary_text=stats_text)
            st.success("AI Batch Summary:")
            st.write(summary)

    # Natural language queries
    st.subheader("💬 Ask Questions About This Dataset")
    query = st.text_input("Example: Show all frauds over $1000")
    if st.button("Run Query with AI"):
        llm = get_llm()
        if llm:
            agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
            with st.spinner("Thinking..."):
                response = agent.run(query)
            st.write(response)

    # Manual per-record explanation (still available)
    st.subheader("🔎 Explain Any Transaction by Index")
    idx = st.number_input("Select row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    if st.button("Explain Selected Transaction"):
        llm = get_llm()
        if llm:
            with st.spinner("Generating explanation..."):
                explanation = explain_with_ai(
                    llm,
                    df.iloc[idx].drop(labels=['Fraud_Probability', 'Fraud_Prediction', 'Fraud_Label']),
                    df.iloc[idx]['Fraud_Probability']
                )
            st.markdown("**📝 Explanation:**")
            st.write(explanation)

else:
    st.info("📥 Please upload a CSV file to start analysis.")
