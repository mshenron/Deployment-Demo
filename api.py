# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
# import xgboost


app = Flask(__name__)

with open('Model/model.pkl','rb') as fp:
		model = pickle.load(fp)
		
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	SL = request.form.get('sepal_length')
	SW = request.form.get('sepal_width')
	PL = request.form.get('petal_length')
	PW = request.form.get('petal_width')
	Input = [[SL,SW,PL,PW]]
	prediction = model.predict(Input)[0]
	return render_template('index.html', OUTPUT=str(prediction))

if __name__ == "__main__":
    app.run(debug=True)