from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------- Load model and scaler safely ----------
# Use relative paths so it works both locally and on Render
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'forest_fire_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    model = None
    scaler = None
    print("Error loading model or scaler:", e)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('index.html', prediction_text="Model or scaler not loaded!", color_class="high-risk")

    try:
        # Feature order must exactly match training
        features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

        # Ensure all inputs exist and are numeric
        input_values = []
        for feat in features:
            value = request.form.get(feat)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing value for {feat}")
            input_values.append(float(value))

        # Convert to DataFrame to match scaler feature names
        input_df = pd.DataFrame([input_values], columns=features)

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Get probability if available
        try:
            probabilities = model.predict_proba(input_scaled)[0]
            classes = model.classes_
            prob_index = list(classes).index(prediction)
            probability = probabilities[prob_index] * 100
        except Exception:
            probability = 0

        # Determine fire risk and message
        high_labels = [1, 'High', 'high', 'HIGH']

        if prediction in high_labels:
            result_text = (
                f"⚠️ WARNING: HIGH FIRE RISK ({probability:.2f}%)<br>"
                "Your forest is in danger!<br>"
                "Take preventive measures immediately.<br>"
                "Ensure humidity levels are maintained and fire control units are alert."
            )
            color_class = "high-risk"
        else:
            result_text = (
                f"✅ LOW FIRE RISK ({probability:.2f}%)<br>"
                "Your forest is safe and the chance of fire is minimal.<br>"
                "Weather conditions appear stable with no major fire threat."
            )
            color_class = "low-risk"

        return render_template('index.html', prediction_text=result_text, color_class=color_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", color_class="high-risk")


# ---------- Main entry ----------
if __name__ == "__main__":
    # Get PORT from environment (Render gives it automatically)
    port = int(os.environ.get("PORT", 5000))
    # Turn off debug for production
    app.run(host="0.0.0.0", port=port, debug=False)
