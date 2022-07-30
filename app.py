from pandas.core.arrays import string_
import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('linearprice.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index2.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    SqFt = float(request.args.get('SqFt'))

    Bedrooms = float(request.args.get('Bedrooms'))

    Batherooms = float(request.args.get('Batherooms'))

    offers = float(request.args.get('offers'))

    Brick =float(request.args.get('Brick'))

    Neighborhood =float(request.args.get('Neighborhood'))
    
    prediction = model.predict([[SqFt,Bedrooms,Batherooms,offers,Brick,Neighborhood]])
        
    return render_template('index2.html', prediction_text='Regression Model  has predicted price for give SqurFeetis{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
