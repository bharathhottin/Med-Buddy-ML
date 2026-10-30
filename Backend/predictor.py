import os
import logging

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from joblib import load

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# load the model only once
model = load(MODEL_PATH)
logging.info("Model Loaded Sucessfully....")

def predict(input_data: dict):

    # create Dataframe with correct column order
    df = pd.DataFrame([input_data])

    # class prediction
    prediction = int(model.predict(df)[0])
    logging.info(f"Prediction by Model => {prediction}")

    # probability prediction (positive class)
    probability = float(model.predict_proba(df)[0][1])
    logging.info(f"Probability given by Model => {probability}")

    return {
        "prediction": prediction,
        "probability": probability
    }

# example usage
# sample_input_data = {
#     "age": 52,
#     "sex": 1,
#     "cp": 0,
#     "trestbps": 125,
#     "chol": 212,
#     "fbs": 0,
#     "restecg": 1,
#     "thalach": 168,
#     "exang": 0,
#     "oldpeak": 1.0,
#     "slope": 2,
#     "ca": 0,
#     "thal": 2
# }

# result = predict(input_data=sample_input_data)
# print(result)