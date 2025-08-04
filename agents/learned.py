import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

MODEL_PATH = "feedback/learned_agent_model.pkl"

# Fallback: dummy model in case not trained yet
class DummyModel:
    def predict(self, X):
        return [1.5 for _ in X]  # neutral score in 0–3 range

try:
    model = joblib.load(MODEL_PATH)
except:
    model = DummyModel()

def extract_features(question: str, answer: str, context: str, agent_outputs: list):
    features = [
        len(answer),
        len(set(answer.split())),
        len(context),
        sum([agent["score"] for agent in agent_outputs]) / len(agent_outputs),
    ]
    return np.array(features).reshape(1, -1)

def grade_learned(question: str, answer: str, context: str, agent_outputs: list):
    feats = extract_features(question, answer, context, agent_outputs)
    predicted = float(model.predict(feats)[0])
    print(f"🧠 Raw learned score: {predicted}")
    return {
        "agent": "learned",
        "score": min(max(predicted, 0), 3),
        "explanation": "Learned model prediction based on prior logs."
    }
    

# Optional: call this to train

def train_learned_agent(log_path="feedback/log.json"):
    X, y = [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                question = entry["question"]
                answer = entry["answer"]
                context = entry["context"]
                true_score = entry["true_score"]
                agent_outputs = entry["agents"]
                
                feat = extract_features(question, answer, context, agent_outputs)
                X.append(feat.flatten())
                y.append(true_score)
            except:
                continue

    if len(X) < 5:
        print("❗ Not enough training data for learned agent.")
        return

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"📊 Learned agent MAE on test set: {mae:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Saved learned agent model to {MODEL_PATH}")
    print("Training on targets:", y)
    print("Sample features:", X[:3])
    print("Model prediction on sample:", model.predict(X[:3]))
