from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests
import io

app = FastAPI()

MODEL_URL = "https://fraudstoreisaac.blob.core.windows.net/models/model.pkl?sp=r&st=2025-08-02T10:44:14Z&se=2025-08-31T18:59:14Z&spr=https&sv=2024-11-04&sr=b&sig=2920ubbCTP0Kphr4QMBocImkB6tQG86zCtAITsb%2Fi00%3D"
SCALER_URL = "https://fraudstoreisaac.blob.core.windows.net/models/scaler.pkl?sp=r&st=2025-08-02T10:47:56Z&se=2025-08-31T19:02:56Z&spr=https&sv=2024-11-04&sr=b&sig=e0O3hlrrK2Vtkz8WFaRX5oGnOe9Do7OydfTrWhn%2FAVM%3D"

# Load model and scaler from Azure Blob Storage
def download_and_load(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

model = download_and_load(MODEL_URL)
scaler = download_and_load(SCALER_URL)

class FraudInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

@app.post("/predict")
def predict(input: FraudInput):
    data = np.array([[input.feature1, input.feature2, input.feature3, input.feature4, input.feature5]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return {"fraud_prediction": int(prediction)}
