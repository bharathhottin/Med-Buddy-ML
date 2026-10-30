from fastapi import FastAPI
from pydantic import BaseModel

from Backend.predictor import predict

app = FastAPI(
    title="Heart Disease Predictor API",
    version="1.0.0"
)

# input schema matching training features
class HeartDiseaseInput(BaseModel):
    age: int
    sex:int
    cp: int
    trestbps: float
    chol: float
    fbs:int
    restecg:int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/health")
def health_check():
    return {"status":"ok"}



# disease prediction endpoint
@app.post("/heart_disease_prediction")
def predict_heart_disease(input_data: HeartDiseaseInput):
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return {
           "prediction": result["prediction"],
           "probability": round(result["probability"],3),
           "diagnosis": (
               "Heart Disease Detected"
               if result["prediction"] == 1
               else "No Disease Detected"
            )
    }
   