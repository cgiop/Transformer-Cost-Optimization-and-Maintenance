# Transformer Cost Optimization and Maintenance AI

How to run Backend: python3 -m uvicorn main2:app --reload
Frontend: streamlit run streamlit_app.py
The Reinforcement Learning (RL) model learns optimal maintenance decisions from historical machine data to reduce downtime and costs. By simulating weekly machine behavior, it maximizes uptime and profit through smart preventive actions.

## 🌟 System Overview

The project consists of two main components that communicate seamlessly:

1. **The FastAPI Backend (`main2.py`)**: The brain of the operation. It handles the core Reinforcement Learning logic, data processing, Q-Learning model training, state normalization, and exposes RESTful API endpoints for prediction. It persists the trained model to `transformer_rl_model.pkl` so you don't have to retrain on every startup.
2. **The Streamlit Frontend (`streamlit_app.py`)**: A sleek, modern user interface. It provides a beautiful dashboard for operators to input current machine telemetry and receive AI-driven maintenance recommendations (using Confidence Bars). Advanced controls (like uploading new CSV datasets and triggering AI retraining) are safely tucked away in an Admin Mode sidebar.

---

## 🚀 Running the Application

To run the full application, you will need to start **both** the backend and the frontend in separate terminal windows.

### Step 1: Start the Backend (FastAPI)

Open your first terminal, navigate to the project folder, and run:

```bash
uvicorn main2:app --reload
```

This starts the backend server at `http://127.0.0.1:8000`. 
*(Optional: You can access the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`)*

### Step 2: Start the Frontend (Streamlit)

Open a **new** terminal window (leaving the backend running in the first one), navigate to the project folder, and run:

```bash
streamlit run streamlit_app.py
```

This command will automatically open the frontend dashboard in your default web browser (usually at `http://localhost:8501`). 

---

## 🔌 Core API Endpoints Reference

If you wish to interact with the backend programmatically (bypassing the Streamlit UI), you can use the following primary endpoints:

1. **`POST /upload_data`**
   - **Description:** Accepts a CSV file via form-data and saves it to the root directory as `Transformer_Data_Merged.csv`, effectively updating the underlying data store.

2. **`POST /train`**
   - **Description:** Reads the historical dataset, performs feature engineering, and trains the Q-learning agent for 500 episodes. It ultimately saves the weights and the state scaler locally as a `.pkl` file.
   - **Usage Note:** You must initiate training at least once (either via the API or the Streamlit Admin sidebar) before making predictions to initialize the model!

3. **`POST /predict_raw`**
   - **Description:** The primary inference endpoint. It accepts raw equipment telemetry (e.g., `Uptime_Percentage`, `MTBF_Hours`) as JSON, internally calculates the 9 engineered state features, normalizes them, and returns expected Q-values alongside the optimal `recommended_action` (`Do Nothing`, `Preventive Maintenance`, or `Corrective Maintenance`).
