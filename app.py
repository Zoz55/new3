from flask import *
from keras.models import load_model
from keras.preprocessing import image
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS

dic = {0 : 'Covid', 1 : 'Healthy', 2 : 'Lung Tumor', 3 : 'Common Pneumonia'}

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(64,64))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 64,64,3)
	p = model.predict_classes(i)
	return dic[p[0]]

app = Flask(__name__,  static_folder="static")
CORS(app)

filename1 = 'chest_model_balanced.h5'
model = keras.models.load_model(filename1)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "This Project Was Designed By The Students of Faculty of Electronic Engineering - Egypt"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['image']
		imagename =  image.filename
		img_path = os.path.join("C:\Users\Fayez\Downloads" +imagename)
		image.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
