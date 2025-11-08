from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import pandas as pd

# Import TensorFlow/Keras with error handling
try:
    from tensorflow import keras
    load_model = keras.models.load_model
except ImportError:
    try:
        import keras
        load_model = keras.models.load_model
    except ImportError:
        print("Error: Neither tensorflow nor keras could be imported")
        load_model = None

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models and preprocessing objects
print("Loading models and preprocessing objects...")
try:
    # Load the extracted features dataframe to get the label encoder
    extracted_df = pd.read_pickle('extracted_df.pkl')
    
    # Recreate the label encoder
    le = LabelEncoder()
    le.fit(extracted_df['class'].values)
    
    # Load trained models
    ann_model = load_model('Model1.h5')
    cnn1d_model = load_model('Model2.h5')
    cnn2d_model = load_model('Model3.h5')
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    ann_model = None
    cnn1d_model = None
    cnn2d_model = None
    le = None


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(file_path):
    """Extract MFCC features from audio file"""
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(file_path, res_type="soxr_hq")
        
        # Extract MFCC features
        feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
        
        # Scale the features
        feature_scaled = np.mean(feature.T, axis=0)
        
        return feature_scaled
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")


def predict_ann(feature):
    """Predict using ANN model"""
    if ann_model is None:
        return None, None
    
    prediction_feature = np.array([feature])
    predicted_vector = np.argmax(ann_model.predict(prediction_feature, verbose=0), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    
    # Get confidence scores
    predictions = ann_model.predict(prediction_feature, verbose=0)[0]
    confidence = float(np.max(predictions) * 100)
    
    return predicted_class[0], confidence


def predict_cnn1d(feature):
    """Predict using CNN1D model"""
    if cnn1d_model is None:
        return None, None
    
    prediction_feature = np.array([feature])
    final_prediction_feature = np.expand_dims(prediction_feature, axis=2)
    predicted_vector = np.argmax(cnn1d_model.predict(final_prediction_feature, verbose=0), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    
    # Get confidence scores
    predictions = cnn1d_model.predict(final_prediction_feature, verbose=0)[0]
    confidence = float(np.max(predictions) * 100)
    
    return predicted_class[0], confidence


def predict_cnn2d(feature):
    """Predict using CNN2D model"""
    if cnn2d_model is None:
        return None, None
    
    prediction_feature = np.array([feature])
    final_prediction_feature = prediction_feature.reshape(prediction_feature.shape[0], 16, 8, 1)
    predicted_vector = np.argmax(cnn2d_model.predict(final_prediction_feature, verbose=0), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    
    # Get confidence scores
    predictions = cnn2d_model.predict(final_prediction_feature, verbose=0)[0]
    confidence = float(np.max(predictions) * 100)
    
    return predicted_class[0], confidence


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'ann': ann_model is not None,
            'cnn1d': cnn1d_model is not None,
            'cnn2d': cnn2d_model is not None
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict audio class from uploaded file"""
    print("\n=== New prediction request ===")
    
    # Check if file is present in request
    if 'file' not in request.files:
        print("Error: No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}")
    
    # Check if file is selected
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        print(f"Error: File type not allowed: {file.filename}")
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get model preference (default to all)
    model_type = request.form.get('model', 'all').lower()
    print(f"Model type: {model_type}")
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Extract features
        print("Extracting features...")
        features = extract_features(filepath)
        print(f"Features extracted: shape {features.shape}")
        
        # Make predictions based on model type
        results = {}
        
        if model_type in ['ann', 'all']:
            print("Predicting with ANN...")
            predicted_class, confidence = predict_ann(features)
            if predicted_class:
                results['ann'] = {
                    'predicted_class': predicted_class,
                    'confidence': round(confidence, 2)
                }
                print(f"ANN prediction: {predicted_class} ({confidence:.2f}%)")
        
        if model_type in ['cnn1d', 'all']:
            print("Predicting with CNN1D...")
            predicted_class, confidence = predict_cnn1d(features)
            if predicted_class:
                results['cnn1d'] = {
                    'predicted_class': predicted_class,
                    'confidence': round(confidence, 2)
                }
                print(f"CNN1D prediction: {predicted_class} ({confidence:.2f}%)")
        
        if model_type in ['cnn2d', 'all']:
            print("Predicting with CNN2D...")
            predicted_class, confidence = predict_cnn2d(features)
            if predicted_class:
                results['cnn2d'] = {
                    'predicted_class': predicted_class,
                    'confidence': round(confidence, 2)
                }
                print(f"CNN2D prediction: {predicted_class} ({confidence:.2f}%)")
        
        # Clean up uploaded file
        os.remove(filepath)
        print("Uploaded file cleaned up")
        
        if not results:
            print("Error: No valid predictions")
            return jsonify({'error': 'No valid predictions could be made'}), 500
        
        response_data = {
            'success': True,
            'filename': filename,
            'predictions': results
        }
        print(f"Returning response: {response_data}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': str(e)}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available classes"""
    if le is None:
        return jsonify({'error': 'Label encoder not loaded'}), 500
    
    return jsonify({
        'classes': le.classes_.tolist()
    })


@app.route('/', methods=['GET'])
def index():
    """Serve the HTML frontend"""
    try:
        with open('main.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            'name': 'Audio Classification API',
            'version': '1.0.0',
            'endpoints': {
                '/health': 'GET - Health check',
                '/predict': 'POST - Predict audio class (requires file upload)',
                '/classes': 'GET - Get list of available classes',
            },
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'models': ['ann', 'cnn1d', 'cnn2d']
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
