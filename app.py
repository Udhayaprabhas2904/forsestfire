from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # ✅ Prevent Tkinter thread errors
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import os
from textwrap import wrap

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ✅ Register Times New Roman font safely
try:
    if os.path.exists("times.ttf"):
        pdfmetrics.registerFont(TTFont('Times-Roman', 'times.ttf'))
except Exception as e:
    print("Font registration failed:", e)

# ✅ Load model and scaler
model = pickle.load(open(r'C:\Users\user\Desktop\forestfire_project\forest_fire_model (2).pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\user\Desktop\forestfire_project\scaler (1).pkl', 'rb'))

# Global variables to store last prediction
last_result = None
last_input = None
last_probability = None


@app.route('/')
def home():
    """Render the main web page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle fire risk prediction and show results"""
    global last_result, last_input, last_probability

    try:
        # ✅ Input features
        features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
        input_values = [float(request.form[feat]) for feat in features]
        input_df = pd.DataFrame([input_values], columns=features)

        # ✅ Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # ✅ Probability handling
        try:
            probabilities = model.predict_proba(input_scaled)[0]
            classes = model.classes_
            prob_index = list(classes).index(prediction)
            probability = probabilities[prob_index] * 100
        except Exception:
            probability = 0

        # ✅ Determine fire risk level
        high_labels = [1, 'High', 'high', 'HIGH']

        if prediction in high_labels or (isinstance(prediction, (int, float)) and prediction == 1):
            result_text = (
                f"HIGH FIRE RISK ({probability:.2f}%) — "
                "Your forest is in danger! Take preventive measures immediately."
            )
            color_class = "high-risk"
        else:
            result_text = (
                f"LOW FIRE RISK ({probability:.2f}%) — "
                "Your forest is safe and stable under current weather conditions."
            )
            color_class = "low-risk"

        # ✅ Save for PDF report
        last_result = result_text
        last_input = dict(zip(features, input_values))
        last_probability = probability

        return render_template('index.html', prediction_text=result_text, color_class=color_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}", color_class="high-risk")


@app.route('/download_pdf')
def download_pdf():
    """Generate and download the PDF report"""
    global last_result, last_input, last_probability

    if not last_result or not last_input:
        return "⚠ No prediction available. Please submit the form first."

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ✅ Page border
    margin = 40
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(3)
    pdf.rect(margin, margin, width - 2 * margin, height - 2 * margin)

    # ✅ Title
    pdf.setFont("Times-Roman", 22)
    pdf.setFillColorRGB(0.1, 0.3, 0.6)
    pdf.drawCentredString(width / 2, height - 70, "FOREST FIRE RISK PREDICTION REPORT")

    # ✅ Decorative line
    pdf.setStrokeColorRGB(0.1, 0.3, 0.6)
    pdf.setLineWidth(1.2)
    pdf.line(margin + 60, height - 80, width - (margin + 60), height - 80)

    # ✅ User Input Table
    y_pos = height - 120
    pdf.setFont("Times-Bold", 13)
    pdf.setFillColor(colors.black)
    pdf.drawString(margin + 20, y_pos, "User Input Details:")

    y_pos -= 25
    pdf.setFont("Times-Bold", 11)
    pdf.setFillColor(colors.lightblue)
    pdf.rect(margin + 20, y_pos - 5, 400, 20, fill=1)
    pdf.setFillColor(colors.black)
    pdf.drawString(margin + 30, y_pos, "Parameter")
    pdf.drawString(margin + 250, y_pos, "Value")
    y_pos -= 25

    toggle = False
    for key, value in last_input.items():
        fill_color = colors.whitesmoke if toggle else colors.lightgrey
        toggle = not toggle
        pdf.setFillColor(fill_color)
        pdf.rect(margin + 20, y_pos - 5, 400, 20, fill=1)
        pdf.setFillColor(colors.black)
        pdf.setFont("Times-Roman", 11)
        pdf.drawString(margin + 30, y_pos, str(key))
        pdf.drawString(margin + 250, y_pos, str(value))
        y_pos -= 21
        if y_pos < 260:
            pdf.showPage()
            y_pos = height - 100

    # ✅ Reduced space before prediction result
    y_pos -= 20

    # ✅ Prediction Result (bold + bigger + aligned)
    text_x = margin + 40
    text_y = y_pos - 25

    pdf.setFont("Times-Bold", 15)
    pdf.setFillColor(colors.black)
    pdf.drawString(text_x, text_y, "Prediction Result:")

    text_y -= 18
    pdf.setFont("Times-Bold", 13)
    wrapped_lines = wrap(last_result, width=70)
    for line in wrapped_lines:
        pdf.drawString(text_x, text_y, line)
        text_y -= 15

    # ✅ Slightly reduced gap before Probability
    text_y -= 8
    pdf.setFont("Times-Bold", 15)
    pdf.drawString(text_x, text_y, "Fire Risk Probability:")
    pdf.setFont("Times-Bold", 13)
    pdf.drawString(text_x + 220, text_y, f"{last_probability:.2f}%")

    # ✅ Fire Risk Graph (closer spacing)
    color_bar = 'red' if "HIGH" in last_result.upper() else 'green'
    fig, ax = plt.subplots(figsize=(3.3, 2))
    ax.bar(['Fire Risk'], [last_probability], color=color_bar)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Fire Risk Probability Level", fontsize=10)
    plt.tight_layout()

    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png', dpi=120)
    plt.close(fig)
    chart_buffer.seek(0)
    chart_image = ImageReader(chart_buffer)

    # ✅ Graph closer to text (reduced spacing)
    chart_y = text_y - 160
    pdf.drawImage(chart_image, text_x, chart_y, width=260, height=150)

    # ✅ Footer
    pdf.setFont("Times-Roman", 10)
    pdf.setFillColor(colors.darkgray)
    pdf.drawCentredString(width / 2, 50, "Report generated by Forest Fire Risk Prediction System")

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Forest_Fire_Risk_Report.pdf",
        mimetype='application/pdf'
    )


if __name__ == "__main__":
    app.run(debug=True)
