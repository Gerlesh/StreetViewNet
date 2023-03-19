import numpy as np
import google_streetview.api

def get_lat_long(x,y):
    # TODO: Get the lat and long from the x and y values
	return (x,y)

if __name__ == "__main__":
	#kmeans_models = np.load("kmeans_models.npy") # You might have to rerun kmeans, k = 22 (or whatever you want). Object arrays lookin wacky
	blue_points = np.load("blue_points_indicies.npz")
	# print(len(blue_points["x"])) # 12 million points lol
	
	for i in range(35000): # Change this to whatever, just make sure you don't overcharge me :(
		idx = np.random.randint(0, len(blue_points["x"]))
		lat,long = get_lat_long(blue_points["x"][idx], blue_points["y"][idx])
			
		params = [{
		# max 640x640 pixels
			'size': '640x640',
			'location': f'{lat},{long}',
			'key': 'AIzaSyA_ilIhZ12IFupmuTPvTEz5GM4bWVgxxaM',
			'radius': "10000"
		}]

		results = google_streetview.api.results(params)
		results.download_links('downloads')  # you can change the directory, not sure how to rename the img though
    
	