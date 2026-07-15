from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/flood_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    prediction = model.predict([values])[0]

    if prediction == 1:
        result = "Flood Likely"
    else:
        result = "No Flood"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)