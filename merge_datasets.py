import os
import shutil
import sys

def rename_files(dir):
	for subdir in os.listdir(dir):
		lines = []
		with open(os.path.join(dir, subdir, "lat_long_images.txt"), 'r') as f:
			lines = f.readlines()
			for filename in os.listdir(os.path.join(dir, subdir)):
				if not filename.endswith('.jpg'):
					continue
				number = int(filename[4:-4])
				file = os.path.join(dir, subdir, filename)
				data = ' '.join(lines[number].split(" ")[:2])
				new_name = os.path.join(sys.argv[1], subdir, data+".jpg")
				if os.path.isfile(new_name):
					print("File already exists: ", new_name)
					os.remove(file)
				else:
					os.rename(file,new_name)

def flatten_datset(dir):
	for subdir in os.listdir(dir):
		for filename in os.listdir(os.path.join(dir, subdir)):
			if not filename.endswith('.jpg'):
				continue
			old_name = os.path.join(dir,subdir,filename)
			new_name = os.path.join(dir,filename)
			if os.path.isfile(new_name):
				print("File already exists: ", new_name)
				os.remove(old_name)
			else:
				os.rename(old_name,new_name)

		shutil.rmtree(os.path.join(dir,subdir))

if __name__=="__main__":
	rename_files(sys.argv[1])
	flatten_datset(sys.argv[1])
