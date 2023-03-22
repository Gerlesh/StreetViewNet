import numpy as np
from argparse import ArgumentParser
import os
import google_streetview.api
from pyproj import Transformer

TRANSFORMER = Transformer.from_crs("EPSG:3857","EPSG:4326")
WIDTH,HEIGHT = 81664, 31776

def get_lat_long(x,y):
    # TODO: Get the lat and long from the x and y values

    lat, long = TRANSFORMER.transform((2*x/WIDTH-1)*20037508,(2*y/HEIGHT-260/335)*20037508*790/2040)
    # Random ratios taken from image measurements
    # [-20037508,20037508] is the range used by EPSG:3857 (Web Mercator Projection standard)
    
    return (lat,long)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--key")
    parser.add_argument("-d","--download",default="dataset")
    parser.add_argument("-N", dest="N", default=1, type=int)

    args = parser.parse_args()

    if not os.path.isdir(args.download):
        os.mkdir(args.download)

    blue_points = np.load("blue_points_indicies.npz")

    idx = np.random.randint(0,len(blue_points["x"]),args.N)
    lats, longs = get_lat_long(blue_points["x"][idx],blue_points["y"][idx])
    headings = np.random.rand(args.N)*360

    params = [{
        'size': '640x640', # max 640x640 pixels
        'location': f'{lat},{long}',
        'key': args.key,
        'heading': str(heading),
        'source': 'outdoor',
        'radius': "1000000"
    } for lat,long,heading in zip(lats,longs,headings)]

    results = google_streetview.api.results(params)

    real_lat = [results.metadata[i]["location"]["lat"] if "location" in results.metadata[i] else str(lats[i]) for i in range(len(results.metadata))]
    real_long = [results.metadata[i]["location"]["lng"] if "location" in results.metadata[i] else str(longs[i]) for i in range(len(results.metadata))]

    results.download_links(args.download)
    N = 0
    for im in results.metadata:
        if im["status"] == "OK":
            N += 1

    print(N,"Images Retreived")

    text = ""
    for i in range(len(real_lat)):
        text += str(real_lat[i])+" "+str(real_long[i])+" "+str(headings[i])+"\n"

    with open(os.path.join(args.download,"lat_long_images.txt"),'w') as f:
        f.write(text)
