# forsestfire
The Forest Fire Risk Prediction project is a Machine Learning-based application designed to predict the likelihood of forest fires based on environmental and meteorological factors.The goal is to analyze fire-prone conditions and provide early warnings to help minimize forest damage, protect wildlife, and assist disaster management authorities.

**Machine Learning Model**
🔹 Algorithms Used
**Random Forest Classifier** – Best accuracy and feature importance visualization
**Logistic Regression** – Baseline performance
**Support Vector Machine (SVM)** – For non-linear data patterns

**Key Features**
**Preprocessing:** Converts categorical month and day into numeric.
**Target:** status → Fire or No Fire.
**Feature Scaling:** Ensures all numeric features are standardized.
**Random Forest Classifier:** Robust for non-linear classification.
**Evaluation:** Accuracy and classification report for training and test sets.
**Persistence:** Saves both model and scaler using pickle.
**Prediction:** Can predict fire risk for any new input.
