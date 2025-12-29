import joblib

class IntentModel:
    def __init__(self, path="mdoel/artifacts/intent_model.pkl"):
        self.pipeline = joblib.load(path)

    def predict(self, text):
        pred = self.pipeline.predict([text])[0]
        probs = self.pipeline.predict_proba([text])[0]
        return {"Intent": pred, "Confidence_Score": max(probs)}
