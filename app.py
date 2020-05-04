# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:09:25 2020

@author: sefa
"""
import pandas as pd
from flask import Flask,jsonify
from sklearn.externals import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/<int:Pclass>/<int:Sex>/<int:Age>/<int:SibSp>/<int:Parch>/<float:Fare>/<string:Embarked>')
def index(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked):
    
    try:     
        if Embarked == "Q":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,1,0]])
        elif Embarked == "C":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,1,0,0]])
        elif Embarked == "S":
            array=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,0,0,1]])
            
        array = scaler.transform(array)    
        print(model.predict(array)[0])    
        print(type(model.predict(array)[0]))          
        return jsonify(result=str(model.predict(array)[0]))
    
    except:
        return jsonify(result=str("error"))
           
if __name__ == '__main__':
    app.run(port=5000,debug=True)
