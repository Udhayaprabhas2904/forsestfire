from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('C:\\Users\\user\\Desktop\\forestfire_project\\forest_fire_model (1).pkl', 'rb'))
scaler = pickle.load(open('C:\\Users\\user\\Desktop\\forestfire_project\\scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from the form
        features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        input_values = [float(request.form[feat]) for feat in features]
        input_data = np.array([input_values])

        # Apply scaler if used during training
        # input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Map probability to predicted class
        classes = model.classes_
        prob_index = list(classes).index(prediction)
        probability = probabilities[prob_index] * 100

        # Determine risk label
        # Handles both numeric (0/1) and string ('Low'/'High') labels
        high_labels = [1, 'High', 'high', 'HIGH']
        if prediction in high_labels:
            result_text = f"HIGH FIRE RISK ({probability:.2f}%)"
            color = "#ff4c4c"
        else:
            result_text = f"LOW FIRE RISK ({probability:.2f}%)"
            color = "#4caf50"

        return render_template('index.html', prediction_text=result_text, color=color)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", color="#ff4c4c")


if __name__ == "__main__":
    app.run(debug=True)
