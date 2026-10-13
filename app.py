import json
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model once, when the app starts
with open("phishing_rf.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

with open("feature_options.json") as f:
    OPTIONS = json.load(f)
with open("examples.json") as f:
    EXAMPLES = json.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    values, result, phish_pct = {}, None, None

    if request.method == "POST":
        # 1. read the 21 form values and convert text -> whole numbers
        values = {name: int(request.form[name]) for name in FEATURES}
        # 2. one-row table, columns in the exact order the model was trained on
        row = pd.DataFrame([values], columns=FEATURES)
        # 3. ask the model
        prediction = model.predict(row)[0]
        result = "Phishing" if prediction == -1 else "Legitimate"
        # 4. roughly the share of the 100 trees that voted "phishing"
        phish_index = list(model.classes_).index(-1)
        phish_pct = round(model.predict_proba(row)[0][phish_index] * 100)
    else:
        # GET request: optionally pre-fill the form with a dataset example
        values = EXAMPLES.get(request.args.get("example"), {})

    return render_template("index.html", features=FEATURES, options=OPTIONS,
                           values=values, result=result, phish_pct=phish_pct)


if __name__ == "__main__":
    app.run(debug=True)
