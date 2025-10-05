from pydantic import BaseModel

class PredictionRequest(BaseModel):
    goals: list[str]
    levels: list[str]
    equipment: list[str]
    program_length: int
    time_per_workout: int

class PredictionResponse(BaseModel):
    program: str
    total_score: float
