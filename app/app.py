import os
from Descriptors.colordescriptor import ColorDescriptor
from Descriptors.texturedescriptor import TextureDescriptor
from Descriptors.shapedescriptor import ShapeDescriptor
from searcher import Searcher
from flask.helpers import flash
from werkzeug.utils import secure_filename
import secrets
from flask import Flask, make_response, render_template, request, redirect
import json
import csv
import time
import numpy as np
import cv2
import shutil

# create flask instance
app = Flask(__name__)
app.secret_key = "secret_key112211"


INDEX_COLOR = os.path.join(os.path.dirname(__file__), 'color.csv')
INDEX_TEXTURE = os.path.join(os.path.dirname(__file__), 'texture.csv')
INDEX_SHAPE = os.path.join(os.path.dirname(__file__), 'shape.csv')


# main route
@app.route('/')
def check():
    if os.path.exists('static/temp') == True :
        shutil.rmtree('static/temp')
        shutil.rmtree('static/upload')
        return redirect('/home')
    else :
        return redirect('/home')


@app.route('/home')
def home():
    datasets = os.listdir('static/101_ObjectCategories/')
    if os.path.exists('static/temp') == True :
        image_names = os.listdir('static/temp')
        nearest = sorted(os.listdir('static/temp'))[0]
        target = os.listdir('static/upload')
        return render_template("index.html", image_names=sorted(image_names,reverse=True),\
        target=(target), aw=1, count=len(datasets), nearest=(nearest))
    else :
        return render_template("index.html", aw=2, count=len(datasets))



@app.route('/search', methods=['POST'])
def search():
    pictures = request.files['image']
    file = pictures.read()
    name = pictures.filename[0:3]
    cd = ColorDescriptor()
    td = TextureDescriptor()
    sd = ShapeDescriptor()
    npimg = np.frombuffer(file, np.uint8)
    query = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    query_gry = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    color_features = cd.describe(query)
    texture_feature = td.lbp(query_gry)
    shape_feature = sd.extractFeatures(query_gry)
  
     
   # perform the search
    searcher = Searcher(INDEX_COLOR,INDEX_TEXTURE,INDEX_SHAPE)
    results = searcher.search(color_features,texture_feature,shape_feature)


    os.makedirs('static/temp')
    os.makedirs('static/upload')
    
    i = 1
    for (score, resultID) in results:
        i += 1
        result = cv2.imread("static/101_ObjectCategories/" + resultID)
        saveimg = cv2.imwrite("static/temp/" + str(score) + str(i) + ".jpeg", result)
    

    imgstr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite("static/upload/"+ imgstr +".jpeg", query)
    return redirect("/home")

@app.route('/<page_name>')
def other_page(page_name):
    response = make_response('The page named %s does not exist.' \
                             % page_name, 404)
    return response

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)