from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('fish_weight_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    prediction_text = f'Predicted Fish Weight: {prediction[0]:.2f}'
    
    return f'The predicted weight of the fish is: {prediction_text} grams'

if __name__ == '__main__':
    app.run(debug=True)
