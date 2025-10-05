# api-service/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Gym Planner API Service")

ML_SERVICE_URL = "http://localhost:8001"

class UserInput(BaseModel):
    goals: list[str]
    levels: list[str]
    equipment: list[str]
    program_length: int
    time_per_workout: int

@app.post("/suggest-programs")
def suggest_programs(user_input: UserInput):
    payload = user_input.model_dump()
    response = requests.post(f"{ML_SERVICE_URL}/predict", json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"ML service returned {response.status_code}: {response.text}"
        }
