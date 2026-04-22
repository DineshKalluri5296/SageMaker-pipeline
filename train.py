import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import boto3

# Load data
data = pd.read_csv("s3://ml-project-buckets/data/USA_Housing (1).csv")

X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file("model.pkl", "ml-project-buckets", "model/model.pkl")

print("Model trained and uploaded")
