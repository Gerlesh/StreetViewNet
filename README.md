# StreetViewNet

Geolocation estimator from Street View Images

## Images Directory

### Latitude Directories
Contains the Google Maps images captured using a macro corresponding to the directory name's latitude.

### cropper.py
Used to crop all the images in the latitude directories, cropped images are saved in their corresponding latitude directory.

### img_appender.py
Appends all cropped images to create a large (~700MB) image of the world map.

## Root

### blue_points.py
Finds all Street View locations on appended map and saves their indices in blue_points_indicies.npz

### k-means.py
Ran k-means with different k, graphed distortion. Currently runs k=22 k-means and saves the model to kmeans22.sav.

### streetview_pics.py
Uses blue_points_indicies.npz, converts (x, y) points to latitude and longitude and uses the API to download images corresponding to a random point in the .npz file into a given directory.

### labeler.py
Labels Street View images using a given k-means model and saves it to a .npz array. The saved files take up a lot of space, might label during training.

## License

[MIT](https://choosealicense.com/licenses/mit/)
