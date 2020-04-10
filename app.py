from flask import Flask, render_template ,request, redirect, url_for, flash, jsonify
import numpy as np
import pickle 
import json


app = Flask(__name__)

model = pickle.load(open('model.pkl' , 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit' , methods=['POST'])
def submit_data():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        
        output = prediction[0]
        
        return render_template('index.html' , prediction_text = 'prediction is {}'.format(output))
       

if __name__ == '__main__':
    app.run(debug=False)