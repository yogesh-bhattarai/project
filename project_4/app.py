from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features = features.split(',')
    np_features = np.asarray(features, dtype=np.float32)

    # prediction
    pred = model.predict(np_features.reshape(1, -1))
    message = ["cancerous" if pred[0] == 1 else "Non-concerous"]
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
