from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from titanic_survival_ml.predict import get_model, make_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked, title):
    family = sibsp + parch
    X_pred = pd.DataFrame({
            "Pclass": pd.Series(pclass, dtype='float64'),
            "Sex": pd.Series(sex, dtype='object'),
            "Age": pd.Series(age, dtype='float64'),
            "SibSp": pd.Series(sibsp, dtype='float64'),
            "Parch": pd.Series(parch, dtype="float64"),
            "Fare": pd.Series(fare, dtype="float64"),
            "Embarked": pd.Series(embarked, dtype='object'),
            "Title": pd.Series(title, dtype='object'),
            "Family": pd.Series(family, dtype="float64")
            })
    result, survival_probability = make_prediction(X_pred)
    if result == 1:
        prediction = 'Survived'
    else:
        prediction = 'Not survived'
    return {"prediction": prediction, 
            "survival_probability": survival_probability}
    