import os
from Descriptors.colordescriptor import ColorDescriptor
from Descriptors.texturedescriptor import TextureDescriptor
from Descriptors.shapedescriptor import ShapeDescriptor
from weight_generator import weight_generator, reset_weights
from searcher import Searcher
from flask.helpers import flash
from werkzeug.utils import secure_filename
import secrets
from flask import Flask, make_response, render_template, request, redirect, url_for, session
import json
import csv
import time
import numpy as np
import cv2
import shutil
from shutil import copyfile

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
        reset_weights() #To reset weights file from old weights
        return redirect('/home')
    else :
        return redirect('/home')


@app.route('/home')
def home():
    if os.path.exists('static/temp') == True :
        image_names = os.listdir('static/temp')
        target = os.listdir('static/upload')
        return render_template("index.html", image_names=sorted(image_names, reverse=True),\
        target=(target), aw=1)
    else :
        return render_template("index.html", aw=2)



@app.route('/search', methods=['POST'])
def search():
    descriptors_list = request.form.getlist('checkedDescriptor')
    pictures = request.files['image']
    file = pictures.read()
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
    results = searcher.search(color_features,texture_feature,shape_feature,descriptors=descriptors_list)


    os.makedirs('static/temp')
    os.makedirs('static/upload')
    old_list = []
    for (score, resultID) in results:
        result = cv2.imread("static/101_ObjectCategories/" + resultID)
        saveimg = cv2.imwrite("static/temp/" + str(score) + ".jpeg", result)
        old_list.append(resultID)
    #We save our list for the images in a object session
    session['old_list'] = old_list
    imgstr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite("static/upload/"+ imgstr +".jpeg", query)

    return redirect("/home")



@app.route('/tryagain', methods=['POST'])
def try_again():
    goodImgs = request.form.getlist('goodimg')
    shutil.rmtree('static/tmp')
    os.makedirs('static/tmp')
    for g in goodImgs:
        copyfile('static/temp/'+g,'static/tmp/'+g)
    shutil.rmtree('static/temp')
    shutil.copytree('static/tmp/','static/temp/')

    target = os.listdir('static/upload')
    img = cv2.imread(os.path.join('static/upload/',target[0]))
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cd = ColorDescriptor()
    td = TextureDescriptor()
    sd = ShapeDescriptor()
    color_features = cd.describe(img)
    texture_feature = td.lbp(img_grey)
    shape_feature = sd.extractFeatures(img_grey)
    # random weights for each descriptor: Color, Texture, Shape
    weights = weight_generator()
    # perform the search with the new weights
    searcher = Searcher(INDEX_COLOR,INDEX_TEXTURE,INDEX_SHAPE)
    results = searcher.search(color_features,texture_feature,shape_feature,limit=20,w=weights)
    print(results)

    for (score, resultID) in results:
        result = cv2.imread("static/101_ObjectCategories/" + resultID)
        if resultID not in session.get('old_list'):
            saveimg = cv2.imwrite("static/temp/" + str(score) + ".jpeg", result)
            session['old_list'].append(resultID)
            session.modified = True
    
      
    return redirect("/home")

@app.route('/<page_name>')
def other_page(page_name):
    response = make_response('This page does not exist.' \
                             , 404)
    return response

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)