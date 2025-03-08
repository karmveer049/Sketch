from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS  # Allows frontend to communicate with backend

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  # Receive features from frontend
    data = np.array(data).reshape(1, -1)  # Convert to numpy array
    scaled_data = scaler.transform(data)  # Scale input data

    prediction = model.predict(scaled_data)[0]  # Get prediction
    return jsonify({"prediction": int(prediction)})  # Return result

if __name__ == "__main__":
    app.run(debug=True)
