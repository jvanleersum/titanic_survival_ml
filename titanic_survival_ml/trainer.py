from titanic_survival_ml.data import get_data, impute_missing_values, set_X_y
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from termcolor import colored

class Trainer():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
    def set_pipeline(self):
        preproc = ColumnTransformer(
            [
        ("min_max", MinMaxScaler(), ["SibSp", "Parch", "Family"]),
        ("robust", RobustScaler(), ["Age", "Fare"]),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), ["Pclass", "Sex", "Embarked", "Title"])
            ]
        )

        self.pipeline = Pipeline(
            [
                ("preproc", preproc),
                ("forest", RandomForestClassifier(n_estimators=200, max_depth=5))
            ]
        )
        return self.pipeline
    
    def run(self):
        self.set_pipeline()
        self.pipeline.fit(X_train, y_train)
        
    def evaluate(self):
        accuracy = self.pipeline.score(self.X_test, self.y_test)
        print("accuracy: ", accuracy)
        return accuracy
    
    def save_model_locally(self):
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        
if __name__ == "__main__":
    df = get_data()
    df = impute_missing_values(df)
    X, y = set_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, X_test, y_train, y_test)
    trainer.run()
    trainer.evaluate()
    trainer.save_model_locally()    
    