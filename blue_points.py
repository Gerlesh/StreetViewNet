import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	blue_values = [[175,158,18], [189,173,49], [180,163,29], [179,162,26], [179,162,25], [185,168,39]]  # BGR
	lower_bound = [175,158,18]
	upper_bound = [189,173,49]
	img = cv2.imread("./Images/appended.png")

	# find all pixels that are within the blue range
	mask = cv2.inRange(img, np.array(lower_bound), np.array(upper_bound))
	indices = np.nonzero(mask == 255)
	indices = np.array(indices)
	indices[0] = np.subtract(mask.shape[0], indices[0])
	plt.plot(indices[1], indices[0], "o", markersize=0.5)
	plt.title("StreetView Locations")
	plt.show()
	np.savez_compressed("blue_points_indicies.npz", x = indices[1], y = indices[0])