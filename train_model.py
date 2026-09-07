import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("simulated_journeys.csv")

features = ["num_transfers", "total_scheduled_time", "min_buffer", "pct_bus", "hour", "day_of_week", "deadline_slack"]
X = df[features]
y = df["reached_on_time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

with open("deadline_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": features}, f)

print("Model saved to deadline_model.pkl")