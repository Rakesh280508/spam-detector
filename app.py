from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    # Transform input
    data = vectorizer.transform([message])

    # Predict
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "🚫 Spam Message"
    else:
        result = "✅ Not Spam Message"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
