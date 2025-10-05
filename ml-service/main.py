from fastapi import FastAPI
from .schemas.prediction import PredictionRequest, PredictionResponse
from .utils.predictor import predict

app = FastAPI(title="ML Service")

@app.post("/predict", response_model=list[PredictionResponse])
def predict_program(req: PredictionRequest):
    results = predict(
        goals=req.goals,
        levels=req.levels,
        equipment=req.equipment,
        program_length=req.program_length,
        time_per_workout=req.time_per_workout
    )
    return results
