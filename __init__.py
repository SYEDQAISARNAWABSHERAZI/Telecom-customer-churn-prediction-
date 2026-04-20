import os
import json
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

from config import METRICS_PATH, FEATURE_IMPORTANCE_CSV, PLOTS_DIR
from src.predict import predict_churn

app = Flask(__name__)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        prediction, probability, risk_level = predict_churn(form_data)

        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )

    except Exception as e:
        return f"Error: {str(e)}"
    

@app.route("/dashboard")
def dashboard():
    metrics = load_metrics()
    chart_files = [
        "churn_distribution.png",
        "contract_vs_churn.png",
        "tenure_vs_churn.png",
        "monthlycharges_vs_churn.png",
        "confusion_matrix.png",
        "roc_curve.png"
    ]
    return render_template("dashboard.html", metrics=metrics, chart_files=chart_files)


@app.route("/feature-importance")
def feature_importance():
    fi = []
    if os.path.exists(FEATURE_IMPORTANCE_CSV):
        df = pd.read_csv(FEATURE_IMPORTANCE_CSV).head(15)
        fi = df.to_dict(orient="records")
    return render_template("feature_importance.html", feature_importance=fi)


@app.route("/plots/<filename>")
def plots(filename):
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)