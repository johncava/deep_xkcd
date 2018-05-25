import cv2
import numpy as np
import sys
import glob

if __name__ == '__main__' :
    comics = []
    with open('./xkcd_archive/list.txt') as f:
        for line in f:
            line = line.split('\n')
            comics.append(line[0])
    comics = comics[-130:]
    dataset = []
    for comic in comics:
        directory = "xkcd_archive/" + str(comic)
        regions = glob.glob(directory + "/regions.data")
        if len(regions) > 0:
            dataset.append(str(comic))
    with open('dataset.txt', 'w') as f:
        for d in dataset:
            f.write(d + '\n')
    print len(dataset)