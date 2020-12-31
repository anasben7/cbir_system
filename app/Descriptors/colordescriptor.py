import numpy as np
import cv2


class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins

	def describe(self,img, gx=1, gy=1):
		# convert the image to the HSV color space and initialize
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# grab the dimensions
		height, width = img.shape[:2]

		# color
		hists = []
		for i in range(gx):
			for j in range(gy):
				grid = img[i * height // gx:(i + 1) * height // gx, j * width // gy:(j + 1) * width // gy]
				# extract a 3D color histogram from the masked region of the
				# image, using the supplied number of bins per channel
				hist = cv2.calcHist([grid], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256], accumulate=False)

				hists.append(hist)

		#normalization
		hists = np.array(hists)
		hists = hists / np.sum(hists)
		# flatten
		hists = hists.flatten()


		return hists
