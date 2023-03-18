import os
from PIL import Image

def crop_image(image_path, coords, saved_location):
	"""
	@image_path: the path to the image to edit
	@coords: a tuple of coordinates (left, upper, right, lower)
	@saved_location: path to save the cropped image
	"""
	image_obj = Image.open(image_path)
	w, h = image_obj.size
	cropped_image = image_obj.crop((coords[0], coords[1], w - coords[2], h - coords[3]))
	cropped_image.save(saved_location)

if __name__ == "__main__":
	#loop through all directories and crop images
	for root, dirs, files in os.walk("."):
		for file in files:
			if file.endswith(".png"):
				file_path = os.path.join(root, file)
				crop_path = os.path.join(root, "cropped_" + file)
				crop_image(file_path, (10, 178, 10, 240), crop_path)