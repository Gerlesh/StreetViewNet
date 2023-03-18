import os
import cv2
import numpy as np

def concat_vh(list_2d):
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

if __name__ == "__main__":
    # loop through all directories and append cropped images
	dir_nums = []
	img_rows = []
	i = 0
	for subdirs, dirs, files in os.walk("."):
		img_row = []
		if len(dirs) != 0:
			for dir in dirs:
				# extract the interger in the directory name
				dir_nums.append(int(dir.split(" ")[0]))
		for file in files:
			if file.startswith("cropped_"):
				image_obj = cv2.imread(os.path.join(subdirs, file))
				img_row.append(image_obj)
		img_rows.append([dir_nums[i], img_row])
		i += 1
	# sort the rows by the directory number
	img_rows.sort(key=lambda x: x[0], reverse=True)
	
	img_rows = np.array(img_rows)
	imgs = img_rows[:, 1]
	lats = img_rows[:, 0]
	print(lats)
	img = concat_vh(imgs)
	cv2.imwrite("appended.png", img)
	np.save("lats.npy", lats)