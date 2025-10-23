from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('C:\\Users\\user\\Desktop\\forestfire_project\\forest_fire_model (2).pkl', 'rb'))
scaler = pickle.load(open('C:\\Users\\user\\Desktop\\forestfire_project\\scaler (1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Feature order must exactly match training
        features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

        # Collect input values from form
        input_values = [float(request.form[feat]) for feat in features]
        input_data = np.array([input_values])

        # Convert to DataFrame to match scaler feature names
        input_df = pd.DataFrame(input_data, columns=features)

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Probability (if model supports)
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
                f"WARNING: HIGH FIRE RISK ({probability:.2f}%)<br>"
                "Your forest is in danger!<br>"
                "Take preventive measures immediately.<br>"
                "Maintain moisture and stay alert."
            )
            color_class = "high-risk"
        else:
            result_text = (
                f"LOW FIRE RISK ({probability:.2f}%)<br>"
                "Your forest is safe and the chance of fire is minimal.<br>"
                "Weather conditions appear stable with no major fire threat."
            )
            color_class = "low-risk"

        return render_template('index.html', prediction_text=result_text, color_class=color_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", color_class="high-risk")


if __name__ == "__main__":
    app.run(debug=True)
