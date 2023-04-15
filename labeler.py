import os
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from pyproj import Transformer
import pickle

TRANSFORMER = Transformer.from_crs("EPSG:4326","EPSG:3857")
WIDTH,HEIGHT = 81664, 31776

def get_x_y(lat,long):
	x, y = TRANSFORMER.transform(lat,long)
	x = (x/20037508+1)*WIDTH/2
	y = (y/20037508/790*2040+260/335)*HEIGHT/2
	return (x,y)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-d","--directory",default="dataset")
	parser.add_argument("-m","--model", default="kmeans22.sav")
	parser.add_argument("-o","--outputdir", default="resized")
	parser.add_argument("-s","--size", default=256, type=int)
	args = parser.parse_args()

	if not os.path.isdir(args.directory) or not os.path.isfile(args.model) or not os.path.isdir(args.outputdir):
		exit("Directory does not exist")

	kmeans = pickle.load(open(args.model, 'rb'))
	labels = np.zeros(kmeans.n_clusters)
	for file in os.listdir(args.directory):
		if file.endswith(".jpg"):
			image = Image.open(os.path.join(args.directory, file))
			new_image = image.resize((args.size, args.size))
			lat_long = file[:-4].split()
			x,y = get_x_y(float(lat_long[0]), float(lat_long[1]))
			label = kmeans.predict(np.array([x,y]).reshape(1,-1))[0]
			new_image.save(os.path.join(args.outputdir, file[:-4]+f" {label}.jpg"))
			labels[label] += 1
	print(labels)


			