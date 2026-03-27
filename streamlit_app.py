import streamlit as st
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Maintenance AI - Original Logic",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background-color: #0f1115;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #1a1d24;
    border-right: 1px solid #2d3748;
}

.main-header {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-align: center;
}

.sub-header {
    color: #94a3b8;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 3rem;
    font-weight: 300;
}

.card-title {
    color: #cbd5e1;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 16px;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
}

[data-testid="stNumberInput"] input {
    background-color: #0f1115 !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 0.75rem !important;
    width: 100%;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3) !important;
}

.result-box {
    border: 1px solid;
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
    animation: slideUp 0.5s ease;
    text-align: center;
}

.result-label {
    color: #94a3b8;
    text-transform: uppercase;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 2px;
}

.result-value {
    font-size: 2.8rem;
    font-weight: 700;
    margin-top: 8px;
}

.confidence-bar {
    margin-top: 16px;
    background: #1e222a;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #2d3748;
}

.cf-label { color: #cbd5e1; font-size: 1rem; font-weight: 500; }
.cf-val { color: #8b5cf6; font-weight: 700; float: right; }

.qval-admin-box {
    background: #0f1115;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px;
    margin-top: 12px;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stFileUploader"] {
    background: #0f1115 !important;
    border: 1px dashed #334155 !important;
}
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE for Q-values (shared between sidebar and main) ---
if "qvals" not in st.session_state:
    st.session_state.qvals = None
if "last_action" not in st.session_state:
    st.session_state.last_action = None


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Admin Controls")
    st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Restricted area for model management and recalibration.</p>", unsafe_allow_html=True)

    admin_toggle = st.toggle("Enable Admin Mode")

    if admin_toggle:
        st.markdown("---")
        st.markdown("#### 1. Database Update")
        st.markdown("<p style='font-size:0.85rem; color:#64748b; line-height:1.3;'>Upload historical machine telemetry logs to expand the system's underlying knowledge base.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV dataset", type="csv", label_visibility="collapsed")

        if st.button("Upload New Data") and uploaded_file is not None:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            with st.spinner("Uploading..."):
                try:
                    resp = requests.post(f"{API_URL}/upload_data", files=files)
                    if resp.status_code == 200:
                        st.success("✅ Data uploaded successfully.")
                    else:
                        st.error("❌ Upload failed.")
                except requests.exceptions.ConnectionError:
                    st.error("Backend server is unreachable.")

        st.markdown("---")
        st.markdown("#### 2. AI Retraining")
        st.markdown("<p style='font-size:0.85rem; color:#64748b; line-height:1.3;'>Recalibrate the predictive maintenance engine based on the latest uploaded dataset.</p>", unsafe_allow_html=True)
        if st.button("Retrain AI Engine"):
            with st.spinner("Calibrating the AI..."):
                try:
                    resp = requests.post(f"{API_URL}/train")
                    if resp.status_code == 200:
                        st.success("✅ Model successfully retrained!")
                    else:
                        st.error("❌ Training failed.")
                except requests.exceptions.ConnectionError:
                    st.error("Backend server is unreachable.")

        # --- Q-VALUES PANEL (Admin Only) ---
        if st.session_state.qvals is not None:
            qv = st.session_state.qvals
            st.markdown("---")
            st.markdown("#### 3. Raw Q-Values")
            st.markdown(f"""
            <div class="qval-admin-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Do Nothing</span>
                    <span style="color:#3b82f6; font-size:1rem; font-weight:700;">{qv[0]:.4f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                    <span style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Preventive</span>
                    <span style="color:#eab308; font-size:1rem; font-weight:700;">{qv[1]:.4f}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px;">Corrective</span>
                    <span style="color:#ef4444; font-size:1rem; font-weight:700;">{qv[2]:.4f}</span>
                </div>
            </div>
            <p style="color:#475569; font-size:0.75rem; margin-top:8px;">Last prediction: <b style="color:#94a3b8;">{st.session_state.last_action}</b></p>
            """, unsafe_allow_html=True)


# --- MAIN UI ---
st.markdown('<div class="main-header">Smart Maintenance AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter the current machine telemetry to receive real-time predictive maintenance actions.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card-title">Operational Health</div>', unsafe_allow_html=True)
    uptime = st.number_input("Uptime Percentage (%)", value=99.5, min_value=0.0, max_value=100.0, step=1.0)
    util_rate = st.number_input("Utilization Rate (%)", value=70.0, min_value=0.0, max_value=100.0, step=1.0)
    sched_h = st.number_input("Scheduled Uptime (Hours)", value=720.0, step=10.0)

with col2:
    st.markdown('<div class="card-title">Failure & Repair History</div>', unsafe_allow_html=True)
    mtbf = st.number_input("Mean Time Between Failures (HT)", value=10000.0, step=50.0)
    mttr = st.number_input("Mean Time To Repair (HT)", value=1.0, step=1.0)
    breakdowns = st.number_input("Recent Breakdowns (Count)", value=0.0, step=1.0)
    downtime = st.number_input("Total Downtime (Hours)", value=1.0, step=1.0)

with col3:
    st.markdown('<div class="card-title">Production & Costs</div>', unsafe_allow_html=True)
    output_qty = st.number_input("Total Output Quantity", value=20000.0, step=500.0)
    reject_qty = st.number_input("Defective/Reject Quantity", value=5.0, step=10.0)
    maint_cost = st.number_input("Maintenance Parts Cost ($)", value=50.0, step=50.0)
    energy_kwh = st.number_input("Energy Consumption (kWh)", value=1000.0, step=50.0)

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    analyze_clicked = st.button("🤖 Analyze & Recommend Action")

if analyze_clicked:
    payload = {
        "Uptime_Percentage": uptime,
        "MTBF_Hours": mtbf,
        "MTTR_Hours": mttr,
        "Utilization_Rate": util_rate,
        "Scheduled_Hours": sched_h,
        "Downtime_Duration": downtime,
        "Maintenance_Parts_Cost": maint_cost,
        "Energy_Consumption_kWh": energy_kwh,
        "Output_Quantity": output_qty,
        "Reject_Quantity": reject_qty,
        "Number_of_Breakdowns": breakdowns
    }

    with st.spinner("Analyzing machine status..."):
        try:
            resp = requests.post(f"{API_URL}/predict_raw", json=payload)
            success = resp.status_code == 200
        except requests.exceptions.ConnectionError:
            success = False
            error_msg = "Could not connect to the AI engine on port 8001. Ensure main2.py is running."

    if success:
        data = resp.json()
        action = data["recommended_action"]
        qvals = data["q_values"]

        # Store in session state so admin sidebar can read it
        st.session_state.qvals = qvals
        st.session_state.last_action = action

        if "Corrective" in action:
            bg_color = "rgba(239, 68, 68, 0.1)"
            border_color = "#ef4444"
            icon = "🔧"
            bar_color_recommended = "#ef4444"
        elif "Preventive" in action:
            bg_color = "rgba(234, 179, 8, 0.1)"
            border_color = "#eab308"
            icon = "🛡️"
            bar_color_recommended = "#eab308"
        else:
            bg_color = "rgba(34, 197, 94, 0.1)"
            border_color = "#22c55e"
            icon = "✅"
            bar_color_recommended = "#22c55e"

        # Result box
        st.markdown(f"""
        <div class="result-box" style="background: {bg_color}; border-color: {border_color}; border-left-width: 6px;">
            <div class="result-label">AI Recommended Action</div>
            <div class="result-value" style="color: {border_color};">{icon} {action}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence: Q(action) / sum(abs(all Q-values))
        q_arr = np.array(qvals)
        abs_sum = np.sum(np.abs(q_arr))
        if abs_sum == 0:
            probs = np.ones(3) / 3.0
        else:
            probs = np.abs(q_arr) / abs_sum

        _, cf_col, _ = st.columns([1, 2, 1])
        with cf_col:
            st.markdown("<div class='confidence-bar'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title' style='margin-bottom:20px; text-align:center;'>Analysis Confidence Levels</div>", unsafe_allow_html=True)

            actions_list = ["Do Nothing", "Preventive Maintenance", "Corrective Maintenance"]
            bar_colors = ["#3b82f6", "#eab308", "#ef4444"]

            for i, (p, name, bar_col) in enumerate(zip(probs, actions_list, bar_colors)):
                is_best = actions_list[i] == action or name == action
                label_style = f"font-weight:700; color:#f1f5f9;" if is_best else ""
                progress_html = f"""
                <div style="margin-bottom: 16px;">
                    <span class='cf-label' style="{label_style}">{name}</span>
                    <span class='cf-val' style="color:{bar_col};">{p*100:.1f}%</span>
                    <div style='background:#334155; height:8px; border-radius:4px; margin-top:8px; width:100%; overflow:hidden;'>
                        <div style='background:{bar_col}; height:100%; width:{p*100:.1f}%; border-radius:4px;'></div>
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        err = resp.text if 'resp' in locals() else error_msg
        st.error(f"Analysis failed. Ensure the AI model has been trained by an administrator. (Error: {err})")
