
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model using a path relative to app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "03_models" / "decision_tree_model.pkl"

model = joblib.load(MODEL_PATH)

# Keep the same feature order as the saved model
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# EXAMPLE mappings: verify every mapping against your cleaning code!
MAPPINGS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},

    "MultipleLines": {
        "No": 0,
        "Yes": 1,
        "No phone service": 2,
    },

    "InternetService": {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2,
    },

    "OnlineSecurity": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },
    "OnlineBackup": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },
    "DeviceProtection": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },
    "TechSupport": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },
    "StreamingTV": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },
    "StreamingMovies": {
        "No": 0,
        "Yes": 1,
        "No internet service": 2,
    },

    "Contract": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    },

    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    },
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer = {}

        # Convert the submitted form values to the model's input format
        for feature in FEATURES:
            value = request.form.get(feature, "").strip()

            if not value:
                return render_template(
                    "index.html",
                    error=f"Please provide {feature}.",
                )

            if feature in MAPPINGS:
                if value not in MAPPINGS[feature]:
                    return render_template(
                        "index.html",
                        error=f"Invalid value for {feature}.",
                    )

                customer[feature] = MAPPINGS[feature][value]

            elif feature in ("MonthlyCharges", "TotalCharges"):
                customer[feature] = float(value)

            else:
                customer[feature] = int(value)

        # Create a DataFrame with the exact expected feature order
        input_data = pd.DataFrame([customer], columns=FEATURES)

        # Generate the prediction: normally 1 = churn, 0 = no churn
        prediction = int(model.predict(input_data)[0])

        return render_template(
            "index.html",
            prediction=prediction,
        )

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values for tenure and charges.",
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Check the model and input format.",
        )


if __name__ == "__main__":
    app.run(debug=True)


