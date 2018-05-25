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
    with open('delete.sh', 'w') as W:
        for comic in comics:
            directory = "./xkcd_archive/" + str(comic)
            regions = glob.glob(directory + "/regions.data")
            if len(regions) < 1:
                continue
            # Read Image
            W.write("cd " + directory + "\n")
            '''
            img_file = glob.glob(directory + "/*.png")
            img = Image.open(directory + "/" + img_file[0].split("/")[-1])
            img = img.resize((512,512), Image.ANTIALIAS)
            img_data = list(img.getdata(band = 0))
            img_data = np.array(img_data)
            img_data = np.reshape(img_data, (512,512))
            '''
            with open(directory + '/regions.data','r') as R:
                count = 1
                for line in R:
                    W.write("rm " + str(count) + ".png\n")
                    count = count + 1
            W.write("cd ../../\n")
            print comic