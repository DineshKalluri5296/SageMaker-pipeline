import joblib
import pandas as pd
from sklearn.metrics import r2_score

# Load model
model = joblib.load("model.pkl")

# Load dataset (same file downloaded in train.py)
data = pd.read_csv("data.csv")

data = data.drop(["Address"], axis=1)

X = data.drop("Price", axis=1)
y = data["Price"]

preds = model.predict(X)

score = r2_score(y, preds)

print("Model Score:", score)

# Fail pipeline if low accuracy
if score < 0.7:
    raise Exception("Model performance is low!")
