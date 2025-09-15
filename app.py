import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import joblib  
import logging
import random
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the pre-trained models and preprocessor
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize variables
preprocessor = None
isolation_forest_model = None
autoencoder_model = None
ae_threshold = None

# Flag to determine if we're using mock predictions
using_mock = False

# Try to load the models
try:
    logger.info("Loading preprocessor...")
    try:
        with open(os.path.join(MODEL_DIR, 'preprocessor_bs.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info("Preprocessor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading preprocessor with pickle: {str(e)}")
        try:
            preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor_bs.pkl'))
            logger.info("Preprocessor loaded successfully with joblib.")
        except Exception as e:
            logger.error(f"Error loading preprocessor with joblib: {str(e)}")
            raise

    logger.info("Loading Isolation Forest model...")
    try:
        with open(os.path.join(MODEL_DIR, 'isolation_forest_model_bs.pkl'), 'rb') as f:
            isolation_forest_model = pickle.load(f)
        logger.info("Isolation Forest model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Isolation Forest model with pickle: {str(e)}")
        try:
            isolation_forest_model = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest_model_bs.pkl'))
            logger.info("Isolation Forest model loaded successfully with joblib.")
        except Exception as e:
            logger.error(f"Error loading Isolation Forest model with joblib: {str(e)}")
            raise

    logger.info("Loading Autoencoder model...")
    try:
        autoencoder_model = keras.models.load_model(os.path.join(MODEL_DIR, 'autoencoder_model_bs.h5'))
        logger.info("Autoencoder model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Autoencoder model: {str(e)}")
        raise

    logger.info("Loading Autoencoder threshold...")
    try:
        with open(os.path.join(MODEL_DIR, 'ae_threshold.pkl'), 'rb') as f:
            ae_threshold = pickle.load(f)
        logger.info("Autoencoder threshold loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Autoencoder threshold with pickle: {str(e)}")
        try:
            ae_threshold = joblib.load(os.path.join(MODEL_DIR, 'ae_threshold.pkl'))
            logger.info("Autoencoder threshold loaded successfully with joblib.")
        except Exception as e:
            logger.error(f"Error loading Autoencoder threshold with joblib: {str(e)}")
            raise

except Exception as e:
    logger.error(f"Failed to load one or more models: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.warning("Switching to mock prediction mode")
    using_mock = True

# Mock threshold for demonstration
mock_threshold = 0.5

def mock_predict(input_data):
    """Generate mock predictions when models can't be loaded."""
    logger.info("Using mock prediction")
    
    # Generate a random MSE value between 0 and 1
    mse = random.uniform(0, 1)
    
    # Determine if transaction is normal or fraud based on MSE and threshold
    if_result = "Normal" if random.random() > 0.3 else "Fraud"
    ae_result = "Normal" if mse <= mock_threshold else "Fraud"
    
    return {
        'input_data': input_data,
        'isolation_forest_prediction': if_result,
        'autoencoder_prediction': ae_result,
        'reconstruction_error': float(mse),
        'threshold': float(mock_threshold),
        'note': 'MOCK PREDICTION: Models could not be loaded properly. This is simulated data.'
    }

@app.route('/')
def index():
    # Check if we're using mock predictions
    if using_mock:
        logger.warning("Serving index page in mock prediction mode")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data from request
        data = request.json
        
        # Create a DataFrame-like dictionary with the input data
        input_data = {
            'amount': [float(data.get('amount', 0))],
            'category': [data.get('category', '')],
            'gender': [data.get('gender', '')],
            'age': [data.get('age', '')]
        }
        
        logger.info(f"Received input data: {input_data}")
        
        # If we're using mock predictions, return mock data
        if using_mock:
            return jsonify(mock_predict(input_data))
        
        # Check if all models are loaded
        if not (preprocessor is not None and 
                isolation_forest_model is not None and 
                autoencoder_model is not None and 
                ae_threshold is not None):
            logger.error("One or more models failed to load but using_mock is False")
            return jsonify({
                'error': 'One or more models failed to load. Please check the server logs.'
            }), 500
        
        # Apply preprocessor to the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction with Isolation Forest model
        if_prediction = isolation_forest_model.predict(processed_data)[0]
        if_result = "Normal" if if_prediction == 1 else "Fraud"
        
        # Make prediction with Autoencoder model
        # First, get the reconstruction
        reconstructed = autoencoder_model.predict(processed_data)
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean(np.power(processed_data - reconstructed, 2), axis=1)[0]
        
        # Compare MSE with threshold
        ae_result = "Normal" if mse <= ae_threshold else "Fraud"
        
        # Prepare the response
        response = {
            'input_data': input_data,
            'isolation_forest_prediction': if_result,
            'autoencoder_prediction': ae_result,
            'reconstruction_error': float(mse),
            'threshold': float(ae_threshold)
        }
        
        logger.info(f"Prediction results: IF={if_result}, AE={ae_result}, MSE={mse}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # If an error occurs, fall back to mock predictions
        if not using_mock:
            logger.warning("Error occurred during prediction. Falling back to mock prediction.")
            try:
                return jsonify(mock_predict(input_data))
            except:
                pass
                
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
