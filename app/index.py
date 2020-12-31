# python3 index.py --dataset 101_ObjectCategories/ --index index.csv

from Descriptors.colordescriptor import ColorDescriptor
from Descriptors.texturedescriptor import TextureDescriptor
from Descriptors.shapedescriptor import ShapeDescriptor
import argparse
import glob
import cv2
import numpy as np


# Construct the argument parser and parse the arguments
# python3 index.py --dataset 101_ObjectCategories/ --index index.csv

# After when we gonna deploy this on our web app we gonna change these argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="lien pour dataset")
ap.add_argument("-i", "--index", required=True,
                help="lien pour index.csv")
args = vars(ap.parse_args())


# Initialize the 3 descriptors
# For Color descriptor; with 8 bins for the Hue channel, 12 bins for the saturation channel,
# and 3 bins for the value channel
cd = ColorDescriptor((8,12,3))
td = TextureDescriptor()
sd = ShapeDescriptor()


output = open(args["index"], "w")  # Output file

# We gonna use glob to loop over images : 
for imagePath in glob.glob(args["dataset"] + "*/*.jpg"):
    # First we need something to identify every picture , let's use the image Path
    imageID = imagePath[imagePath.find("/") + 1:]
    image = cv2.imread(imagePath)
    image_gry = cv2.imread(imagePath,0)
    # Now we gonna use our descriptors function to get the features :
    features = cd.describe(image)
    features = np.concatenate([features, td.lbp(image_gry)])
    features = np.concatenate([features, sd.extractFeatures(image_gry)])

    # write and the features into a a csv file .
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close