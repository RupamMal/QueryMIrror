from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import pickle
import numpy as np

# Add parent directory to path to import helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WEB.helper import query_point_creator

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# Load model and vectorizer
try:
    model = pickle.load(open(os.path.join(BASE_DIR, 'WEB', 'model.pkl'), 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/api/check', methods=['POST'])
def check_duplicate():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        q1 = data.get('q1', '').strip()
        q2 = data.get('q2', '').strip()
        
        if not q1 or not q2:
            return jsonify({"error": "Both q1 and q2 are required"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
