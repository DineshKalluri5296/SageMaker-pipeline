import joblib

def model_fn(model_dir):
    return joblib.load(f"{model_dir}/model.pkl")

def predict_fn(input_data, model):
    return model.predict(input_data)
