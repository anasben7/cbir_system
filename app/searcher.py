import numpy as np
import csv

class Searcher:
	def __init__(self,indexPath_color,indexPath_texture,indexPath_shape):
		# Store our index Paths
		self.indexPath_color = indexPath_color
		self.indexPath_texture = indexPath_texture
		self.indexPath_shape = indexPath_shape

	def search(self,queryFeatures_color,queryFeatures_texture,queryFeatures_shape,limit=10,descriptors=['Color','Texture','Shape'],w=[1/3,1/3,1/3]):
		# Initialize the output dictionary
		results = {}

		with open(self.indexPath_color) as  f1,open(self.indexPath_texture) as  f2,open(self.indexPath_shape) as  f3 :
			# Read the csv Files
			reader_color = csv.reader(f1) 
			reader_texture = csv.reader(f2)
			reader_shape = csv.reader(f3)

			# Loop over rows
			for (row_color,row_texture,row_shape) in zip(reader_color,reader_texture,reader_shape):
				total_descriptors = 0
				for descriptor in descriptors:
					if descriptor == 'Color':
						features_color = [float(x) for x in row_color[1:]]
						# For each row we gonna extract the features associated with the indexed
						# image and then Compare them with the given image using the chi2_distance
						descriptor_color = self.chi_squared_distance(features_color, queryFeatures_color)
						total_descriptors += descriptor_color * w[0]
					if descriptor == 'Texture':
						features_texture = [float(x) for x in row_texture[1:]]
						descriptor_texture = self.chi_squared_distance(features_texture, queryFeatures_texture)
						total_descriptors += descriptor_texture * w[1]
					if descriptor == 'Shape':
						features_shape = [float(x) for x in row_shape[1:]]
						descriptor_shape = self.chi_squared_distance(features_shape, queryFeatures_shape)
						total_descriptors += descriptor_shape * w[2]
				# Now after computing the distance of the current feature in X image
				# with our given image we need to save that in our dictionary
				# The key here is the current image id and the value is the distance
				# (How similar the images)
				results[row_color[0]] = total_descriptors  # Here row_color can be changed by texture or just the id of current picture

		f1.close()
		f2.close()
		f3.close()
		# Now we need to sort our list of results
		# ( the low values are the most similar pictures)
		results = sorted([(v,k) for (k,v) in results.items()])
		# Here we return only one result with the lowest value
		return results[:limit]

	# Compute the chi-squared distance
	def chi_squared_distance(self, histA, histB, eps=1e-10):
		d = 0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])
		return d