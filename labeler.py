import os
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from pyproj import Transformer
import re
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
	parser.add_argument("-o","--output", default="dataset.npz")
	args = parser.parse_args()

	if not os.path.isdir(args.directory) or not os.path.isfile(args.model):
		exit("Directory does not exist")

	lat_longs = np.loadtxt(os.path.join(args.directory,"lat_long_images.txt"))

	kmeans = pickle.load(open(args.model, 'rb'))

	imgs = []
	labels = []
	headings = []
	for file in os.listdir(args.directory):
		if file.startswith("gsv_"):
			lat_long = lat_longs[int(re.search(r"\d+", file).group(0))]
			x,y = get_x_y(lat_long[0], lat_long[1])
			label = kmeans.predict(np.array([x,y]).reshape(1,-1))
			im = np.array(Image.open(os.path.join(args.directory, file)))
			imgs.append(im)
			labels.append(label[0])
			headings.append(lat_long[2])
	np.savez_compressed(os.path.join(args.directory,args.output), imgs=imgs, labels=labels, headings=headings)


			