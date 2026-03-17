import joblib
import pandas as pd
import numpy as np
import os

ARTIFACT_DIR = "artifacts"
model_path = os.path.join(ARTIFACT_DIR, "xgb_cybersecurity_model_gpu.pkl")
le_target_path = os.path.join(ARTIFACT_DIR, "label_encoder_target.pkl")
le_attack_path = os.path.join(ARTIFACT_DIR, "label_encoder_attack_type.pkl")
le_high_path = os.path.join(ARTIFACT_DIR, "label_encoders_high_cardinality.pkl")

print(" Loading model and encoders...")
model = joblib.load(model_path)          
le_target = joblib.load(le_target_path)
le_attack = joblib.load(le_attack_path)
le_high = joblib.load(le_high_path)
print(" Artifacts loaded successfully.\n")

user_input = "Zero-Day Exploit,Cloud Service,57.161.159.140,213.142.125.206,48.99,120,Firewall,External User,Germany,7,Finance,87,Quarantine"

columns = [
    "attack_type", "target_system", "attacker_ip", "target_ip", "data_compromised_GB",
    "attack_duration_min", "security_tools_used", "user_role", "location",
    "attack_severity", "industry", "response_time_min", "mitigation_method"
]

values = [x.strip() for x in user_input.split(",")]
if len(values) != len(columns):
    print(f" Expected {len(columns)} values but got {len(values)}.")

input_df = pd.DataFrame([values], columns=columns)

num_cols = ["data_compromised_GB", "attack_duration_min", "attack_severity", "response_time_min"]
input_df[num_cols] = input_df[num_cols].astype(float)

for c in ["attacker_ip", "target_ip"]:
    if c in input_df.columns:
        input_df = input_df.drop(columns=[c])

input_df["data_per_min"] = input_df["data_compromised_GB"] / (input_df["attack_duration_min"] + 1)
input_df["response_efficiency"] = input_df["response_time_min"] / (input_df["attack_duration_min"] + 1)
input_df["severity_ratio"] = input_df["attack_severity"] / (input_df["response_time_min"] + 1)
input_df["is_long_attack"] = (input_df["attack_duration_min"] > 300).astype(int)
input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

input_df["attack_type"] = le_attack.transform(input_df["attack_type"])

for col, enc in le_high.items():
    if col in input_df.columns:
        input_df[col] = input_df[col].map(lambda x: x if x in enc.classes_ else enc.classes_[0])
        input_df[col] = enc.transform(input_df[col])

print(" Making prediction using full pipeline...")
pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0]

pred_label = le_target.inverse_transform([int(pred)])[0]
confidence = float(np.max(prob)) * 100

print("\n Prediction Result:")
print(f" Predicted Outcome: {pred_label}")
print(f" Confidence: {confidence:.2f}%")
print("\n Prediction complete.")
