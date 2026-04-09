import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
import shutil
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

app = FastAPI(title="Transformer RL + Cost Optimization API")

# ---------------- GLOBAL MODEL ----------------
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
            print("✅ Model loaded")
    except:
        print("❌ Model load failed")

# ---------------- INPUT ----------------
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

# ---------------- UPLOAD ----------------
@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    path = "Transformer_Data_Merged.csv"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Dataset uploaded"}

# ---------------- TRAIN ----------------
@app.post("/train")
def train_model():
    path = "Transformer_Data_Merged.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset missing")

    df = pd.read_csv(path).fillna(0)

    # -------- FEATURE ENGINEERING --------
    df["Scheduled_Hours"] = df["Scheduled_Hours"].replace(0, 1)
    df["MTBF_Hours"] = df["MTBF_Hours"].replace(0, 1)
    df["MTTR_Hours"] = df["MTTR_Hours"].replace(0, 1)

    df["Failure_Rate"] = 1 / df["MTBF_Hours"]
    df["Repair_Efficiency"] = 1 / df["MTTR_Hours"]
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

    features = [
        "Health_Index","Failure_Rate","Repair_Efficiency",
        "Utilization_Stress","Downtime_Ratio",
        "Maintenance_Cost_Rate","Energy_Cost_Rate",
        "Reject_Rate","Number_of_Breakdowns"
    ]

    states = df[features].values

    scaler = MinMaxScaler()
    states = scaler.fit_transform(states)

    # ✅ FIX: keep df alignment
    X_train, X_test, df_train, df_test = train_test_split(
        states, df, test_size=0.2, random_state=42
    )

    weights = np.zeros((3, X_train.shape[1]))

    alpha = 0.01
    gamma = 0.95
    epsilon = 0.2

    # -------- ENVIRONMENT --------
    def apply_probabilistic_change(value, prob_inc, prob_dec, step=0.05):
        r = np.random.rand()
        if r < prob_inc:
            value += step
        elif r < prob_inc + prob_dec:
            value -= step
        return value

    def environment_step(state, action):
        next_state = state.copy()
        HEALTH, FAILURE, DOWNTIME, COST = 0, 1, 4, 5

        if action == 0:
            next_state[HEALTH] = apply_probabilistic_change(next_state[HEALTH], 0.1, 0.6)
            next_state[FAILURE] = apply_probabilistic_change(next_state[FAILURE], 0.7, 0.1)
            next_state[DOWNTIME] = apply_probabilistic_change(next_state[DOWNTIME], 0.6, 0.1)
            next_state[COST] += 0.05

        elif action == 1:
            next_state[HEALTH] = apply_probabilistic_change(next_state[HEALTH], 0.7, 0.1)
            next_state[FAILURE] = apply_probabilistic_change(next_state[FAILURE], 0.1, 0.7)
            next_state[DOWNTIME] = apply_probabilistic_change(next_state[DOWNTIME], 0.1, 0.6)
            next_state[COST] -= 0.03

        elif action == 2:
            if np.random.rand() < 0.9:
                next_state[HEALTH] = 0.9
                next_state[FAILURE] = 0.1
                next_state[DOWNTIME] = 0.1
            else:
                next_state[HEALTH] = 0.6
                next_state[FAILURE] = 0.3
                next_state[DOWNTIME] = 0.3
            next_state[COST] += 0.08

        next_state += np.random.normal(0, 0.01, len(state))
        return np.clip(next_state, 0, 1)

    # -------- REWARD (UPDATED) --------
    def compute_reward(state, action, row):
        health = state[0]
        downtime = state[4]
        cost_feature = state[5]

        revenue = row["Output_Quantity"] * (1 - state[7]) * 50
        maintenance_cost = cost_feature * 1000
        downtime_cost = downtime * 2000

        profit = revenue - maintenance_cost - downtime_cost

        action_cost = 500 if action == 1 else 2000 if action == 2 else 0

        return (
            0.5 * profit +
            30 * health -
            40 * downtime -
            50 * cost_feature -
            action_cost
        )

    # -------- TRAIN LOOP --------
    for ep in range(500):
        idx = np.random.randint(len(X_train))
        state = X_train[idx]
        row = df_train.iloc[idx]

        for _ in range(50):
            if np.random.rand() < epsilon:
                action = np.random.randint(3)
            else:
                action = np.argmax(np.dot(weights, state))

            next_state = environment_step(state, action)

            # ✅ FIX: use next_state
            reward = compute_reward(next_state, action, row)

            td = reward + gamma * np.max(np.dot(weights, next_state)) - np.dot(weights[action], state)
            weights[action] += alpha * td * state

            state = next_state

            if state[0] < 0.2:
                break

    model_state["weights"] = weights
    model_state["scaler"] = scaler
    model_state["is_trained"] = True

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"weights": weights, "scaler": scaler}, f)

    return {"message": "Model trained successfully"}

# ---------------- PREDICT ----------------
@app.post("/predict_raw")
def predict(data: RawDataInput):
    if not model_state["is_trained"]:
        raise HTTPException(status_code=400, detail="Train model first")

    sched = data.Scheduled_Hours or 1

    failure = 1/(data.MTBF_Hours or 1)
    repair = 1/(data.MTTR_Hours or 1)
    util = data.Utilization_Rate * sched
    downtime = data.Downtime_Duration / sched

    maint = data.Maintenance_Parts_Cost/(sched+1)
    energy = data.Energy_Consumption_kWh/(data.Output_Quantity+1)
    reject = data.Reject_Quantity/(data.Output_Quantity+1)

    health = 0.35*data.Uptime_Percentage + 0.25*(1-failure) + 0.2*(1-downtime) + 0.2*(1-reject)

    state = np.array([[health,failure,repair,util,downtime,maint,energy,reject,data.Number_of_Breakdowns]])
    scaled = model_state["scaler"].transform(state)[0]

    q = np.dot(model_state["weights"], scaled)
    action_idx = np.argmax(q)

    actions = ["Do Nothing","Preventive Maintenance","Corrective Maintenance"]

    # -------- COST + PROFIT --------
    cost = data.Maintenance_Parts_Cost + data.Energy_Consumption_kWh*0.1 + data.Downtime_Duration*20
    revenue = data.Output_Quantity * 50
    profit = revenue - cost

    return {
        "recommended_action": actions[action_idx],
        "q_values": q.tolist(),
        "estimated_cost": round(cost,2),
        "expected_profit": round(profit,2)
    }