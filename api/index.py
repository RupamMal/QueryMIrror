from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import pickle
import numpy as np
import traceback

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')
WEB_DIR = os.path.join(BASE_DIR, 'WEB')

# Add paths
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, WEB_DIR)

# Import helper after path setup
try:
    from WEB.helper import query_point_creator
except ImportError as e:
    print(f"Error importing helper: {e}")
    query_point_creator = None

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# Load model and vectorizer
model = None
try:
    model_path = os.path.join(WEB_DIR, 'model.pkl')
    # If model missing, attempt to download from environment variable MODEL_URL
    if not os.path.exists(model_path):
        model_url = os.getenv('MODEL_URL')
        if model_url:
            try:
                import requests
                print(f"Model not found locally; downloading from MODEL_URL: {model_url}")
                resp = requests.get(model_url, timeout=30)
                resp.raise_for_status()
                with open(model_path, 'wb') as mf:
                    mf.write(resp.content)
                print(f"Downloaded model to {model_path}")
            except Exception as de:
                print(f"Failed to download model from MODEL_URL: {de}")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
        except Exception as le:
            print(f"Error unpickling model at {model_path}: {le}")
            traceback.print_exc()
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

@app.route('/')
def home():
    try:
        return send_from_directory(PUBLIC_DIR, 'index.html')
    except Exception as e:
        return jsonify({
            "status": "success",
            "message": "Duplicate Question Pair Detection API",
            "endpoints": {
                "check": "/api/check",
                "health": "/api/health"
            }
        })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "helper_loaded": query_point_creator is not None
    })

@app.route('/api/check', methods=['POST'])
def check_duplicate():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        q1 = data.get('q1', '').strip()
        q2 = data.get('q2', '').strip()
        
        if not q1 or not q2:
            return jsonify({"error": "Both q1 and q2 are required"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if query_point_creator is None:
            return jsonify({"error": "Helper functions not loaded"}), 500
        
        # Create query point
        query = query_point_creator(q1, q2)
        
        # Make prediction
        result = model.predict(query)[0]
        probability = model.predict_proba(query)[0]
        
        return jsonify({
            "q1": q1,
            "q2": q2,
            "is_duplicate": bool(result),
            "probability_not_duplicate": float(probability[0]),
            "probability_duplicate": float(probability[1])
        })
    
    except Exception as e:
        print(f"Error in check_duplicate: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        "status": "API is working",
        "base_dir": BASE_DIR,
        "web_dir": WEB_DIR,
        "public_dir": PUBLIC_DIR,
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    app.run(debug=False)
