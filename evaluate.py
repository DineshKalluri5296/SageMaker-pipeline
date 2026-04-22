import joblib
import pandas as pd
from sklearn.metrics import r2_score

model = joblib.load("model.pkl")

data = pd.read_csv("data.csv")

X = data.drop("price", axis=1)
y = data["price"]

preds = model.predict(X)

score = r2_score(y, preds)

print("Model Score:", score)

# Fail pipeline if low accuracy
if score < 0.7:
    raise Exception("Model performance is low!")
