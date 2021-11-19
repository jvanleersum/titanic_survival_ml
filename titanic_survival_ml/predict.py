import joblib

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def make_prediction(X_new):
    model = get_model('model.joblib')
    prediction = model.predict(X_new)
    survival_probability = model.predict_proba(X_new)[1]
    return prediction, survival_probability