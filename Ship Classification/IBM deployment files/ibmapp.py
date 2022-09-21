import re
import numpy as np
import os
from flask import Flask, app, request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat

from keras.applications.vgg16 import preprocess_input

# Loading the model
model = load_model(r"shipclassification.h5")

app = Flask(__name__)


# default home page or route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')


@app.route('/index.html')
def home():
    return render_template("index.html")


@app.route('/result', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)  # getting the current path i.e where app.py is present
        # print("current path",basepath)
        filepath = os.path.join(basepath, 'uploads',
                                f.filename)  # from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        # print("upload folder is",filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))

        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        # reshape data for the model

        pred = model.predict(img)
        # print(pred)
        pred = pred.flatten()
        pred = list(pred)
        m = max(pred)
        # print(pred)
        # print(pred.index(m))
        val_dict = {0: 'Cargo', 1: 'Carrier', 2: 'Cruise', 3: 'Military', 4: 'Tankers'}
        # print(val_dict[pred.index(m)])

        result = val_dict[pred.index(m)]
        print(result)
        return render_template('prediction.html', prediction=result)


""" Running our application """
if __name__ == "__main__":
    app.run()