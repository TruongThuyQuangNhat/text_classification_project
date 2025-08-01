from flask import Flask, request, jsonify
import joblib
import re
from pyvi import ViTokenizer
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://customer-request-classifier.onrender.com"}})
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ViTokenizer.tokenize(text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '')
    processed_text = preprocess(raw_text)
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)[0]
    return jsonify({'label': prediction})

port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)

