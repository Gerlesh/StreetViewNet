import numpy as np
from argparse import ArgumentParser
import pathlib
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
    parser.add_argument("-d","--download",default=".",type=pathlib.Path)
    parser.add_argument("-N", dest="N", default=1, type=int)

    args = parser.parse_args()

    if not args.download.is_dir():
        raise ValueError("Invalid download directory")

    blue_points = np.load("blue_points_indicies.npz")

    lat, long = [],[]
    idx = np.random.randint(0,len(blue_points["x"]),args.N)
    lats, longs = get_lat_long(blue_points["x"][idx],blue_points["y"][idx])
    print(lats,longs)
    params = [{
        'size': '640x640', # max 640x640 pixels
        'location': f'{lat},{long}',
        'key': args.key,
        'radius': "50000"
    } for lat,long in zip(lats,longs)]

    results = google_streetview.api.results(params)
    print(results.metadata)
    results.download_links(args.download)
