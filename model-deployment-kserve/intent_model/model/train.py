import os, joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

X = [
    # original samples
    "hi",
    "hello",
    "how to reset password",
    "cancel my subscription",
    "great service",

    # greetings
    "hey",
    "good morning",
    "good evening",
    "hi there",
    "hello team",
    "hey support",

    # questions
    "how can i change my password",
    "where can i update my profile",
    "how do i contact support",
    "what is the refund policy",
    "can i upgrade my plan",
    "how to download my data",

    # complaints
    "my account is locked",
    "i am not able to login",
    "your service is very slow",
    "i am unhappy with the support",
    "the app keeps crashing",
    "billing issue with my account",

    # praise
    "excellent customer support",
    "really happy with the service",
    "thanks for the quick help",
    "amazing experience",
    "love the new update"
]

y = [
    # original labels
    "greeting",
    "greeting",
    "question",
    "complaint",
    "praise",

    # greetings
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",

    # questions
    "question",
    "question",
    "question",
    "question",
    "question",
    "question",

    # complaints
    "complaint",
    "complaint",
    "complaint",
    "complaint",
    "complaint",
    "complaint",

    # praise
    "praise",
    "praise",
    "praise",
    "praise",
    "praise"
]

pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])
pipeline.fit(X, y)

os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(pipeline, "model/artifacts/intent_model.pkl")
print("Model training completed and saved to 'model/artifacts/intent_model.pkl'")
