import cv2
import numpy as np
import sys
import glob
from PIL import Image

if __name__ == '__main__' :
    comics = []
    with open('./xkcd_archive/list.txt') as f:
        for line in f:
            line = line.split('\n')
            comics.append(line[0])
    comics = comics[-130:]
    dataset = []
    for comic in comics:
        directory = "./xkcd_archive/" + str(comic)
        regions = glob.glob(directory + "/regions.data")
        if len(regions) < 1:
            continue
        # Read Image
        img_file = glob.glob(directory + "/*.png")
        img = Image.open(directory + "/" + img_file[0].split("/")[-1])
        img = img.resize((512,512), Image.ANTIALIAS)
        img_data = list(img.getdata(band = 0))
        img_data = np.array(img_data)
        img_data = np.reshape(img_data, (512,512))
        with open(directory + '/regions.data','r') as R:
            count = 1
            for line in R:
                r = line.strip("\n").split(",")
                r = [int(x) for x in r]
                region_data = img_data[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                region_data = Image.fromarray(region_data.astype('uint8'))
                region_data.save(directory + "/" + str(count) + '.png')
                count = count + 1
        print comic