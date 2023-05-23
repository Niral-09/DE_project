import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from NLP_model import predict_sentiment
from image_prediction import extract_text_from_image
import speech_recognition as sr


app = Flask(__name__)

regmodel = pickle.load(open('classmodel.pkl','rb'))
ps = pickle.load(open('porter.pkl','rb'))

@app.route('/')
def base():
    return render_template("base.html")

@app.route('/txt')
def home():
    return render_template("index.html")

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
    output = predict_sentiment(data)
    if output[0]==0:
        return {
            "result" : "Comment is Toxic."
        }
    if output[0]==1:
        return {
            "result" : "Comment is not Toxic."
        } 

@app.route('/predict',methods= ['POST'])
def predict():
    data = [x for x in request.form.values()]
    print(data[0])
    output = predict_sentiment(data[0])
    if output[0]==0:
        return render_template("false.html")
    if output[0]==1:
        return render_template("true.html")

@app.route('/image_predict_api',methods= ['POST'])
def img_predict():
    if 'image' not in request.files:
        print('No image file provided')

    image_file = request.files['image']
    image_file.save("image1")
    
    data = extract_text_from_image("image1")
    print(data)
    output = predict_sentiment(data)
    if output[0]==0:
        return {
            "result" : "Comment is Toxic."
        }
    if output[0]==1:
        return {
            "result" : "Comment is not Toxic."
        }

@app.route('/img')
def img():
    return render_template("image_base.html")

@app.route('/image_predict',methods= ['POST'])
def imag_predict():
    if 'image' not in request.files:
        return 'No image file provided'

    image_file = request.files['image']
    image_file.save("image1")
    
    data = extract_text_from_image("image1")
    print(data)
    output = predict_sentiment(data)
    if output[0]==0:
        return render_template("false.html")
    if output[0]==1:
        return render_template("true.html")

@app.route('/audio')
def aud():
    return render_template("audio_base.html")

@app.route('/upload',methods= ['POST'])
def aud_predict():
    audio_transcript = request.form.get('audio')
    print('Transcript:', audio_transcript)
    data =  audio_transcript
    print(data)
    output = predict_sentiment(data)
    if output[0]==0:
        return render_template("false.html")
    if output[0]==1:
        return render_template("true.html")

if __name__  ==  "__main__":
    app.run(debug=True)
