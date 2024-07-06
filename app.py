from flask import Flask, jsonify, request
import joblib
import pandas as pd

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load the trained model from .pkl file
model = joblib.load('crop_prediction_model.pkl')

# Endpoint to predict crop based on input data
@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        # Get data from request
        data = request.json
        input_data = pd.DataFrame(data, index=[0])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Prepare response
        response = {'crop_predicted': prediction}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Health check endpoint
@app.route('/')
def index():
    return "Server is up and running!"


@app.route('/predict')
def get_message():
    return "API ready to predict crops change (method:POST)"

if __name__ == '__main__':
    app.run(debug=True)
