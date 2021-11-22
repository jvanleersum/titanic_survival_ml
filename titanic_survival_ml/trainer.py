from titanic_survival_ml.data import get_train_data, impute_missing_values, set_X_y, get_test_data
from titanic_survival_ml.predict import create_kaggle_submission
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
from termcolor import colored

class Trainer():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        # self.X_test = X_test
        self.y_train = y_train
        # self.y_test = y_test
        
        
    def set_pipeline(self):
        num_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("min_max", MinMaxScaler())
            ]
        )
        cat_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(missing_values = None, strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
        preproc = ColumnTransformer(
            [
                ("num_transformer", num_transformer, make_column_selector(dtype_include=["int64", "float64"])),
                ("cat_transformer", cat_transformer, make_column_selector(dtype_include=["object", "bool"]))
            ]
        )
        self.pipeline = Pipeline(
            [
                ("preproc", preproc),
                ("forest", RandomForestClassifier(n_estimators=200, max_depth=4))
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
        print(colored("model_joblib saved locally", "green"))
        
if __name__ == "__main__":
    df_train = get_train_data()
    df_test = get_test_data()
    df_train = impute_missing_values(df_train)
    X_train, y_train = set_X_y(df_train)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.save_model_locally()    
    create_kaggle_submission()