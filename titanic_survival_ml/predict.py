import joblib
import pandas as pd

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def make_prediction(X_new):
    model = get_model('model.joblib')
    prediction = model.predict(X_new)[0]
    survival_probability = model.predict_proba(X_new)[0][1]
    return prediction, survival_probability

def create_kaggle_submission():
    df = pd.read_csv("raw_data/test.csv")
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df["Family"] = df["SibSp"] + df["Parch"]
    X = df.drop(columns=["Cabin", "PassengerId", "Ticket", "Name"])
    model = get_model('model.joblib')
    y_pred = model.predict(X)
    predictions = pd.DataFrame({"PassengerId": df["PassengerId"], 
                                "Survived": y_pred})
    predictions.to_csv("titanic_survival_ml/data/kaggle_submission.csv", index=False)
    
    