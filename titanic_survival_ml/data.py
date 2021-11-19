import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def get_data():
    df = pd.read_csv("../raw_data/train.csv")
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
    df["Family"] = df["SibSp"] + df["Parch"]
    return df

def impute_missing_values(df):
    # Imputing missing values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df["Age"] = imp_mean.fit_transform(df[["Age"]])
    df["Embarked"].fillna("S", inplace=True)
    return df
    
def set_X_y(df):
    X = df.drop(columns=["Survived", "Cabin", "PassengerId", "Ticket", "Name"])
    y = df["Survived"]
    return X, y
    