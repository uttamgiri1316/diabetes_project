from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
application = Flask(__name__)
app = application

# Load scaler and model
try:
    scaler = pickle.load(open(r"D:\pregrad1\august_batch_project\Model\standardScaler.pkl", "rb"))
    model = pickle.load(open(r"D:\pregrad1\august_batch_project\Model\modelForPrediction.pkl", "rb"))
except FileNotFoundError as e:
    raise FileNotFoundError("Make sure the scaler and model files exist in the specified path.") from e

@app.route('/')
def index():
    return render_template('index.html')  # Home page with input form

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract form inputs and ensure they are converted to the appropriate type
            Pregnancies = int(request.form.get("Pregnancies"))
            Glucose = float(request.form.get("Glucose"))
            BloodPressure = float(request.form.get("BloodPressure"))
            SkinThickness = float(request.form.get("SkinThickness"))
            Insulin = float(request.form.get("Insulin"))
            BMI = float(request.form.get("BMI"))
            DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
            Age = float(request.form.get("Age"))

            # Combine inputs into a single array for prediction
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

            # Preprocess the input data using the loaded scaler
            scaled_data = scaler.transform(input_data)

            # Use the model to predict
            prediction = model.predict(scaled_data)

            # Map prediction to result
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

            # Render the result in the output page
            return render_template('single_prediction.html', result=result)

        except ValueError:
            return render_template('error.html', error="Invalid input. Please enter valid data.")
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('home.html')  # Redirect to home if GET request

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
