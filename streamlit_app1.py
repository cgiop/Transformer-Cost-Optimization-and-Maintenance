import streamlit as st
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Maintenance AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------
if "qvals" not in st.session_state:
    st.session_state.qvals = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Admin Controls")

    admin_toggle = st.toggle("Enable Admin Mode")

    if admin_toggle:
        st.markdown("---")

        # ---------- DATA UPLOAD ----------
        st.markdown("#### 📂 Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")

        if uploaded_file:
            st.info(f"Selected file: {uploaded_file.name}")

        if st.button("Upload Data"):
            if uploaded_file is None:
                st.warning("Please select a CSV file first.")
            else:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    resp = requests.post(f"{API_URL}/upload_data", files=files)
                    if resp.status_code == 200:
                        st.success("✅ Dataset uploaded successfully!")
                    else:
                        st.error("❌ Upload failed.")
                except:
                    st.error("❌ Backend not running.")

        # ---------- TRAIN ----------
        st.markdown("---")
        st.markdown("#### 🤖 Retrain Model")

        if st.button("Train / Retrain Model"):
            with st.spinner("Training RL model..."):
                try:
                    resp = requests.post(f"{API_URL}/train")
                    if resp.status_code == 200:
                        st.success("✅ Model trained successfully!")
                        st.session_state.model_trained = True
                    else:
                        st.error("❌ Training failed.")
                except:
                    st.error("❌ Backend not reachable.")

        # ---------- Q VALUES ----------
        if st.session_state.qvals is not None:
            qv = st.session_state.qvals
            st.markdown("---")
            st.markdown("#### 📊 Q Values")

            st.write(f"Do Nothing: {qv[0]:.4f}")
            st.write(f"Preventive: {qv[1]:.4f}")
            st.write(f"Corrective: {qv[2]:.4f}")


# ---------------- MAIN UI ----------------
st.title("⚡ Smart Maintenance AI")
st.caption("Predictive Maintenance using Reinforcement Learning")

col1, col2, col3 = st.columns(3)

with col1:
    uptime = st.number_input("Uptime (%)", 0.0, 100.0, 99.5)
    util = st.number_input("Utilization (%)", 0.0, 100.0, 70.0)
    sched = st.number_input("Scheduled Hours", value=720.0)

with col2:
    mtbf = st.number_input("MTBF", value=10000.0)
    mttr = st.number_input("MTTR", value=1.0)
    downtime = st.number_input("Downtime Hours", value=1.0)
    breakdowns = st.number_input("Breakdowns", value=0.0)

with col3:
    output = st.number_input("Output Quantity", value=20000.0)
    reject = st.number_input("Reject Quantity", value=5.0)
    maint_cost = st.number_input("Maintenance Cost", value=50.0)
    energy = st.number_input("Energy (kWh)", value=1000.0)

# ---------------- PREDICT ----------------
if st.button("🚀 Analyze"):

    payload = {
        "Uptime_Percentage": uptime,
        "MTBF_Hours": mtbf,
        "MTTR_Hours": mttr,
        "Utilization_Rate": util,
        "Scheduled_Hours": sched,
        "Downtime_Duration": downtime,
        "Maintenance_Parts_Cost": maint_cost,
        "Energy_Consumption_kWh": energy,
        "Output_Quantity": output,
        "Reject_Quantity": reject,
        "Number_of_Breakdowns": breakdowns
    }

    try:
        resp = requests.post(f"{API_URL}/predict_raw", json=payload)

        if resp.status_code != 200:
            st.error("❌ Model not trained yet. Train first.")
        else:
            data = resp.json()

            action = data["recommended_action"]
            qvals = data["q_values"]
            cost = data["estimated_cost"]
            profit = data["expected_profit"]

            st.session_state.qvals = qvals
            st.session_state.last_action = action

            # ---------- RESULT ----------
            st.success(f"✅ Recommended Action: {action}")

            # ---------- COST ----------
            c1, c2 = st.columns(2)
            c1.metric("Estimated Cost", f"${cost}")
            c2.metric("Expected Profit", f"${profit}")

            # ---------- CONFIDENCE ----------
            q_arr = np.array(qvals)
            probs = np.abs(q_arr) / np.sum(np.abs(q_arr)) if np.sum(np.abs(q_arr)) != 0 else np.ones(3)/3

            st.subheader("Confidence Levels")

            actions = ["Do Nothing", "Preventive", "Corrective"]
            for i, p in enumerate(probs):
                st.progress(float(p))
                st.write(f"{actions[i]}: {p*100:.2f}%")

    except:
        st.error("❌ Backend not reachable. Start FastAPI server.")