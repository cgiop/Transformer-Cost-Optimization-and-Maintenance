import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
import shutil
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

app = FastAPI(title="Transformer Cost Optimization API - Original Logic")

# Global variables to hold model state
model_state = {
    "weights": None,
    "scaler": None,
    "is_trained": False
}

MODEL_PATH = "transformer_rl_model2.pkl"

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            model_state["weights"] = data["weights"]
            model_state["scaler"] = data["scaler"]
            model_state["is_trained"] = True
            print("Loaded trained model from disk.")
    except Exception as e:
        print(f"Failed to load model from disk: {e}")

class FeaturesInput(BaseModel):
    Health_Index: float
    Failure_Rate: float
    Repair_Efficiency: float
    Utilization_Stress: float
    Downtime_Ratio: float
    Maintenance_Cost_Rate: float
    Energy_Cost_Rate: float
    Reject_Rate: float
    Number_of_Breakdowns: float

class RawDataInput(BaseModel):
    Uptime_Percentage: float
    MTBF_Hours: float
    MTTR_Hours: float
    Utilization_Rate: float
    Scheduled_Hours: float
    Downtime_Duration: float
    Maintenance_Parts_Cost: float
    Energy_Consumption_kWh: float
    Output_Quantity: float
    Reject_Quantity: float
    Number_of_Breakdowns: float

@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    upload_path = "Transformer_Data_Merged.csv"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "message": f"File '{file.filename}' uploaded and saved as {upload_path}. You can now call /train."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

@app.post("/train")
def train_model():
    data_path = "Transformer_Data_Merged.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data file '{data_path}' not found.")
        
    df = pd.read_csv(data_path)
    df = df.fillna(0)

    df["Scheduled_Hours"] = df["Scheduled_Hours"].replace(0, 1)
    df["MTBF_Hours"] = df["MTBF_Hours"].replace(0, 1)
    df["MTTR_Hours"] = df["MTTR_Hours"].replace(0, 1)

    df["Failure_Rate"] = 1.0 / df["MTBF_Hours"]
    df["Repair_Efficiency"] = 1.0 / df["MTTR_Hours"]
    df["Utilization_Stress"] = df["Utilization_Rate"] * df["Scheduled_Hours"]
    df["Downtime_Ratio"] = df["Downtime_Duration"] / df["Scheduled_Hours"]

    df["Maintenance_Cost_Rate"] = df["Maintenance_Parts_Cost"] / (df["Scheduled_Hours"] + 1)
    df["Energy_Cost_Rate"] = df["Energy_Consumption_kWh"] / (df["Output_Quantity"] + 1)
    df["Reject_Rate"] = df["Reject_Quantity"] / (df["Output_Quantity"] + 1)

    df["Health_Index"] = (
        0.35 * df["Uptime_Percentage"] +
        0.25 * (1 - df["Failure_Rate"]) +
        0.20 * (1 - df["Downtime_Ratio"]) +
        0.20 * (1 - df["Reject_Rate"])
    )

    state_features = [
        "Health_Index", "Failure_Rate", "Repair_Efficiency",
        "Utilization_Stress", "Downtime_Ratio",
        "Maintenance_Cost_Rate", "Energy_Cost_Rate",
        "Reject_Rate", "Number_of_Breakdowns"
    ]

    states = df[state_features].values
    
    scaler = MinMaxScaler()
    states_scaled = scaler.fit_transform(states)
    
    X_train, X_test = train_test_split(states_scaled, test_size=0.2, random_state=42)

    num_actions = 3
    state_dim = X_train.shape[1]

    weights = np.zeros((num_actions, state_dim))

    alpha = 0.01
    gamma = 0.95
    epsilon = 0.2
    episodes = 500
    
    def get_q_values(state, w):
        return np.dot(w, state)

    def choose_action(state, w):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        return np.argmax(get_q_values(state, w))

    def apply_probabilistic_change(value, prob_inc, prob_dec, step=0.05):
        r = np.random.rand()
        if r < prob_inc:
            value += step
        elif r < prob_inc + prob_dec:
            value -= step
        return value

    def environment_step(state, action):
        next_state = state.copy()
        HEALTH, FAILURE, DOWNTIME = 0, 1, 4

        if action == 0:  # Do nothing
            next_state[HEALTH] = apply_probabilistic_change(next_state[HEALTH], 0.1, 0.6)
            next_state[FAILURE] = apply_probabilistic_change(next_state[FAILURE], 0.7, 0.1)
            next_state[DOWNTIME] = apply_probabilistic_change(next_state[DOWNTIME], 0.6, 0.1)

        elif action == 1:  # Preventive
            next_state[HEALTH] = apply_probabilistic_change(next_state[HEALTH], 0.7, 0.1)
            next_state[FAILURE] = apply_probabilistic_change(next_state[FAILURE], 0.1, 0.7)
            next_state[DOWNTIME] = apply_probabilistic_change(next_state[DOWNTIME], 0.1, 0.6)

        elif action == 2:  # Corrective
            if np.random.rand() < 0.9:
                next_state[HEALTH] = 0.9
                next_state[FAILURE] = 0.1
                next_state[DOWNTIME] = 0.1
            else:
                next_state[HEALTH] = 0.6
                next_state[FAILURE] = 0.3
                next_state[DOWNTIME] = 0.3

            # minor noise clipping according to notebook
        next_state += np.random.normal(0, 0.01, size=len(state))
        next_state = np.clip(next_state, 0, 1)
        return next_state

    def compute_reward(state, action):
        health = state[0]
        downtime = state[4]

        reward = 100 * health - 80 * downtime

        if action == 1:
            reward -= 20
        elif action == 2:
            reward -= 50

        return reward

    for ep in range(episodes):
        state = X_train[np.random.randint(len(X_train))]
        total_reward = 0

        for step in range(50):
            action = choose_action(state, weights)

            next_state = environment_step(state, action)
            reward = compute_reward(state, action)

            q_current = np.dot(weights[action], state)
            q_next = np.max(get_q_values(next_state, weights))

            td_target = reward + gamma * q_next
            td_error = td_target - q_current

            weights[action] += alpha * td_error * state

            state = next_state
            total_reward += reward
            
    model_state["weights"] = weights
    model_state["scaler"] = scaler
    model_state["is_trained"] = True
    
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"weights": weights, "scaler": scaler}, f)
    except Exception as e:
        print(f"Failed to save model to disk: {e}")
    
    return {"status": "success", "message": f"Model trained for {episodes} episodes using original logic and saved to disk."}

@app.post("/predict_features")
def predict_action_features(features: FeaturesInput):
    if not model_state["is_trained"] or model_state["weights"] is None or model_state["scaler"] is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet. Call /train first.")
        
    state_arr = np.array([[
        features.Health_Index,
        features.Failure_Rate,
        features.Repair_Efficiency,
        features.Utilization_Stress,
        features.Downtime_Ratio,
        features.Maintenance_Cost_Rate,
        features.Energy_Cost_Rate,
        features.Reject_Rate,
        features.Number_of_Breakdowns
    ]])
    
    scaler = model_state["scaler"]
    weights = model_state["weights"]
    
    state_scaled = scaler.transform(state_arr)[0]
    # No bias term array appending here
    
    q_vals = np.dot(weights, state_scaled)
    best_action = int(np.argmax(q_vals))
    
    action_map = {
        0: "Do Nothing",
        1: "Preventive Maintenance",
        2: "Corrective Maintenance"
    }
    
    return {
        "q_values": q_vals.tolist(),
        "best_action_index": best_action,
        "recommended_action": action_map[best_action]
    }

@app.post("/predict_raw")
def predict_action_raw(data: RawDataInput):
    if not model_state["is_trained"] or model_state["weights"] is None or model_state["scaler"] is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet. Call /train first.")
        
    sched_hours = data.Scheduled_Hours if data.Scheduled_Hours != 0 else 1
    mtbf = data.MTBF_Hours if data.MTBF_Hours != 0 else 1
    mttr = data.MTTR_Hours if data.MTTR_Hours != 0 else 1
    
    failure_rate = 1.0 / mtbf
    repair_efficiency = 1.0 / mttr
    utilization_stress = data.Utilization_Rate * sched_hours
    downtime_ratio = data.Downtime_Duration / sched_hours
    maintenance_cost_rate = data.Maintenance_Parts_Cost / (sched_hours + 1)
    energy_cost_rate = data.Energy_Consumption_kWh / (data.Output_Quantity + 1)
    reject_rate = data.Reject_Quantity / (data.Output_Quantity + 1)
    
    health_index = (
        0.35 * data.Uptime_Percentage +
        0.25 * (1 - failure_rate) +
        0.20 * (1 - downtime_ratio) +
        0.20 * (1 - reject_rate)
    )
    
    state_arr = np.array([[
        health_index,
        failure_rate,
        repair_efficiency,
        utilization_stress,
        downtime_ratio,
        maintenance_cost_rate,
        energy_cost_rate,
        reject_rate,
        data.Number_of_Breakdowns
    ]])
    
    scaler = model_state["scaler"]
    weights = model_state["weights"]
    
    state_scaled = scaler.transform(state_arr)[0]
    # No bias term
    
    q_vals = np.dot(weights, state_scaled)
    best_action = int(np.argmax(q_vals))
    
    action_map = {
        0: "Do Nothing",
        1: "Preventive Maintenance",
        2: "Corrective Maintenance"
    }
    
    return {
        "q_values": q_vals.tolist(),
        "best_action_index": best_action,
        "recommended_action": action_map[best_action]
    }
