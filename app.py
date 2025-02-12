from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np
import logging
app = Flask(__name__)
CORS(app)  
logging.basicConfig(level=logging.INFO)
try:
    with open("loan_approval_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {str(e)}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Financial Decision API is running!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        if "income" not in data or "credit_score" not in data:
            return jsonify({"error": "Missing fields: income, credit_score"}), 400

        try:
            income = float(data["income"])
            credit_score = float(data["credit_score"])
        except ValueError:
            return jsonify({"error": "Invalid input format"}), 400

        if income < 0 or credit_score < 300 or credit_score > 850:
            return jsonify({"error": "Invalid values: income must be positive, creditScore between 300-850"}), 400

        # Prepare model input
        input_features = np.array([[income, credit_score]])

        # Make prediction
        prediction = model.predict(input_features)

        # Convert prediction result
        result = int(prediction[0])

        return jsonify({"prediction": result}), 200

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask apps
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)