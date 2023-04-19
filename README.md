# StreetViewNet

Geolocation estimator from Street View Images

## Root

### Data Collection
#### streetview_pics.py
Uses blue_points_indicies.npz, converts (x, y) points to latitude and longitude and uses the API to download images corresponding to a random point in the .npz file into a given directory.

#### merge_datasets.py
Renames Street View images gathered from API and puts them all in one directory.

#### blue_points.py
Finds all Street View locations on appended map (Images directory) and saves their indices in blue_points_indicies.npz.

#### k-means.py
Ran k-means with different k, graphed distortion. Currently runs k=22 k-means and saves the model to kmeans22.sav.

#### labeler.py
Resizes and labels Street View images using a given k-means model. The files are renamed to include the label number at the end of their name.

#### countries.npy
A numpy array including index (label) to readable cluster name (location).

### Model

#### ResNet_Model.py

Training and hyperparameter tuning of ResNet model. Takes in 256x256 images with naming convention "[lat] [long] [label].jpg". Saves the model that achieves the highest validation accuracy as 'net.pt'.

#### test.py

Computes validation accuracy using previous data and hyperparameters. Prints top-1, top-3 and top-5 accuracies.

#### analysis.py

Loads in a test set to check accuracy per class and highest prediction accuracy for each class. CPU only.

## Images Directory

### Latitude Directories
Contains the Google Maps images captured using a macro corresponding to the directory name's latitude.

### cropper.py
Used to crop all the images in the latitude directories, cropped images are saved in their corresponding latitude directory.

### img_appender.py
Appends all cropped images to create a large (~700MB) image of the world map.

## License

BSD 3-Clause License