# 🏋️‍♂️ Gym-PTP MVP

## Gym-PTP (Personal Training Plan) Recommender System

This is a **Minimal Viable Product (MVP)** application developed for the **Intelligent Systems** course as part of **Master Academic Studies** at the **Faculty of Electronic Engineering, University of Niš**.  

The application recommends **personalized gym training programs** based on user profile, fitness goals, experience level, available equipment, desired workout program duration, and desired workout duration per workout. The system uses **classical ML models** to find the most relevant programs for user. The best model is selected automatically and served through a FastAPI backend, with a React frontend for user interaction.

## Dataset
Dataset used in this project:
- https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset

Once you've cloned the repository and downloaded the dataset files, create a new folder called `data` in the root directory of the repository. This folder will store all the CSV files.

## Starting backend
Run following command from root folder: 
```
uvicorn backend.app.main:app --reload
```

Server will work on: http://127.0.0.1:8000/docs where you can test API.