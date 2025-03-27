from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load model and encoder
with open("season_model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        month = int(request.form['month'])
        temperature = float(request.form['temperature'])
        
        # Prediction
        features = np.array([[month, temperature]])
        prediction = model.predict(features)[0]
        season = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'season': season})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
