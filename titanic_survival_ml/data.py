import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def get_train_data():
    df = pd.read_csv("raw_data/train.csv")
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
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

def get_test_data():
    df = pd.read_csv("raw_data/test.csv")
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df["Family"] = df["SibSp"] + df["Parch"]
    return df
    