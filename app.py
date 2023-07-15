#import required packages
from flask import Flask, render_template, request
from flask.helpers import flash
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

#create a Flask object
app = Flask("health-app")


#load the ml model which we have saved earlier in .pkl format



#define the route(basically url) to which we need to send http request
#HTTP GET request method
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes_details')
def diabetes_details():
    return render_template('diabetes_details.html')

@app.route('/heart_details')
def heart_details():
    return render_template('heart_details.html')

@app.route('/cancer_details')
def cancer_details():
    return render_template('cancer_details.html')

@app.route('/kidney_details')
def kidney_details():
    return render_template('kidney_details.html')
    
@app.route('/predict_cancer', methods=['GET','POST'])
def predict_cancer():
    if request.method == 'POST':
     model_cancer = pickle.load(open('breast_cancer.pkl', 'rb'))
     try:
        
        tex = float(request.form['texture_mean'])
        par = float(request.form['perimeter_mean'])
        smooth = float(request.form['smoothness_mean'])
        compact = float(request.form['compactness_mean'])
        sym = float(request.form['symmetry_mean'])
        
        prediction = model_cancer.predict([[tex,par,smooth,compact,sym]])
        if prediction == 1:
            return render_template('new.html',prediction_text="Sorry, you are cancerous!")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('new.html',prediction_text="You are non-cancerous")
        
     except ValueError:
            return render_template('new.html',prediction_text="Please fill the approriate values!")
                

       
    else:
        return render_template('cancer.html')



@app.route('/predict_diabetes', methods=['GET','POST'])
def predict_diabetes():
    if request.method == 'POST':
     model_diabetes = pickle.load(open('diabetes.pkl', 'rb'))

     try:
        
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        
        
        prediction = model_diabetes.predict([[Pregnancies, Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if prediction == 1:
            return render_template('new.html',prediction_text="Sorry, you have diabetes!")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('new.html',prediction_text="You don't have diabetes")
        
     except ValueError:
            return render_template('new.html',prediction_text="Please fill the approriate values!")
                

       
    else:
        return render_template('diabetes.html')




@app.route('/predict_heart', methods=['GET','POST'])
def predict_heart():
    if request.method == 'POST':
     model_heart = pickle.load(open('heart.pkl', 'rb'))

     try:
      

        age= float(request.form['age'])
        sex = int(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol  = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca  = float(request.form['ca'])
        thal  =float(request.form['thal'])
        
        
        
        prediction = model_heart.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if prediction == 1:
            return render_template('new.html',prediction_text="Sorry, you have heart disease!")
        
        #condition for prediction when values are valid
        if prediction==0:
            return render_template('new.html',prediction_text="You don't have heart-disease!")
        
     except ValueError:
            return render_template('new.html',prediction_text="Please fill the approriate values!")

    else:
         return render_template('heart.html')


@app.route('/predict_kidney', methods=['GET','POST'])
def predict_kidney():
    if request.method == 'POST':
     model_diabetes = pickle.load(open('kidney.pkl', 'rb'))

     try:
        
      
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = (request.form['rbc'])
        pc = (request.form['pc'])
        pcc = (request.form['pcc'])
        ba = (request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        pot = float(request.form['pot'])
        wc = float(request.form['wc'])
        htn = (request.form['htn'])
        dm = (request.form['dm'])
        cad = (request.form['cad'])
        pe = (request.form['pe'])
        ane = (request.form['ane'])
        
       
        
        prediction = model_diabetes.predict([[age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc,pot, wc, htn, dm, cad, pe, ane]])
        if prediction == 1:
            return render_template('new.html',prediction_text="Sorry, you have disease!")
        
        #condition for prediction when values are valid
        if prediction== 0:
            return render_template('new.html',prediction_text="You don't have disease")
        
     except ValueError:
            return render_template('new.html',prediction_text="Please fill the approriate values!")
                

       
    else:
        return render_template('kidney.html')


                
if __name__=="__main__":
    #run method starts our web service
    #Debug : as soon as I save anything in my structure, server should start again
    app.run(debug=True)