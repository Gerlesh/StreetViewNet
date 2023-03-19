from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	loaded = np.load('blue_points_indicies.npz')
	x = loaded["x"]
	y = loaded["y"]
	X = np.stack((x, y), axis=1)

	kmeans_models = []
	
	'''
	scores = []
	num_clusters = [5,10,20,50,100,150,200]
	for i in range(len(num_clusters)):
		print(i)
		kmeans_models.append(KMeans(n_clusters=num_clusters[i], random_state=0).fit(X))
		scores.append(kmeans_models[i].score(X))
	# plt.figure(1)
	plt.plot(num_clusters, scores)
	plt.show() ''' # around 20 is the best number of clusters

	for i in range(20,26):
		kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
		plt.figure(i)
		plt.title("K-Means Clusters: " + str(i))
		plt.plot(x,y, "o", markersize=0.5)
		plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], "o", markersize=5)
		plt.show()
		kmeans_models.append(kmeans)
	np.save("kmeans_models.npy", kmeans_models)

	