from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parents[1] / "tone_model" / "tone_model.pkl"
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    prediction = model.predict([req.text])[0]
    return {"tone": prediction}
