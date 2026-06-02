import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset
data = {
    "text": [
        "Win money now",
        "Claim your prize",
        "Hello friend",
        "Let's meet tomorrow",
        "Free gift offer",
        "Important update",
        "Congratulations you won lottery",
        "Call me later",
        "Urgent! click this link",
        "How are you?"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Convert text into numerical form
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = MultinomialNB()
model.fit(X, df["label"])

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")
