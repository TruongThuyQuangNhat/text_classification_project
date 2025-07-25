from flask import Flask, request, jsonify
import joblib
import re
from pyvi import ViTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
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

if __name__ == '__main__':
    app.run(debug=True)
