from flask import Flask, render_template, request, redirect, url_for, jsonify
import re
import pandas as pd
import numpy as np
import cv2
import base64
import joblib

app = Flask(__name__)

def predict_fit(image_data):
    model = joblib.load("joblib_RL_Model.pkl")
    img = base64.b64decode(image_data)
    arr = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    arr = np.asarray(gray)
    arr = arr.flatten()
    arr = arr/255.0
    arr = arr.reshape(-1, 28, 28, 1)
    #print(arr.shape)
    results = model.predict(arr)
    result=np.argmax(results, axis=1)
    res=results[0][result]
    return result[0], res[0]




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_base64 = request.values['imageBase64']
        #print("predict" + img_base64)
        image_data = re.sub('^data:image/.+;base64,', '', img_base64)

        digit, accuracy = predict_fit(image_data)
        results = {'digit': int(digit),
                   'accuracy': float(accuracy)}
        return jsonify(results)
    else:
        return "Predict"


if __name__ == '__main__':
    app.run()
