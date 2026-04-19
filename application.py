import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/ridge.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, form_data=None)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", prediction=None, form_data=None)

    try:
        values = [float(request.form[feat]) for feat in FEATURES]
        scaled = scaler.transform([values])
        result = model.predict(scaled)[0]
        prediction = round(float(result), 2)
        return render_template(
            "index.html",
            prediction=prediction,
            form_data=request.form,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction=True,
            form_data=request.form,
            error=f"Error: {str(e)}",
        )


if __name__ == "__main__":
    app.run(debug=True)
