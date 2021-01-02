# python3 index.py --dataset 101_ObjectCategories/ --index index.csv

from Descriptors.colordescriptor import ColorDescriptor
from Descriptors.texturedescriptor import TextureDescriptor
from Descriptors.shapedescriptor import ShapeDescriptor
import argparse
import glob
import cv2
import numpy as np

# Construct the argument parser and parse the arguments
# python3 index.py --dataset 101_ObjectCategories/ --color color.csv --texture texture.csv --shape shape.csv

# After when we gonna deploy this on our web app we gonna change these argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path for dataset")
ap.add_argument("-c","--color",required = True,
                help = "Path to save color descriptor features")
ap.add_argument("-t","--texture",required = True,
                help = "Path to save texture descriptor features")
ap.add_argument("-s","--shape",required = True,
                help = "Path to save shape descriptor features")
args = vars(ap.parse_args())

# Initialize the 3 descriptors
# For Color descriptor; with 8 bins for the Hue channel, 12 bins for the saturation channel,
# and 3 bins for the value channel

cd = ColorDescriptor()
td = TextureDescriptor()
sd = ShapeDescriptor()

#Output file for every descritpor
output_color = open(args["color"],"w")
output_texture = open(args["texture"],"w")
output_shape = open(args["shape"],"w")

# We gonna use glob to loop over images : 
for imagePath in glob.glob(args["dataset"] + "*/*.jpg"):
    # First we need something to identify every picture , let's use the image Path
    imageID = imagePath[imagePath.find("/") + 1:]
    image = cv2.imread(imagePath)
    img_gry = cv2.imread(imagePath,0)

    # Now we gonna use our descriptors functions to get the features :
    color_features = cd.describe(image)
    texture_features = td.lbp(img_gry)
    shape_features = sd.extractFeatures(img_gry)

    # write and the features into a a csv file .
    color_features = [str(f) for f in color_features]
    texture_features = [str(f) for f in texture_features]
    shape_features = [str(f) for f in shape_features]

    output_color.write("%s,%s\n" % (imageID, ",".join(color_features)))
    output_texture.write("%s,%s\n" % (imageID, ",".join(texture_features)))
    output_shape.write("%s,%s\n" % (imageID, ",".join(shape_features)))


output_color.close
output_texture.close
output_shape.close