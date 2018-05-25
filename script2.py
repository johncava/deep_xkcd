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
    '''
    # Read image
    comic = sys.argv[1]
    directory = "xkcd_archive/" + str(comic)
    img_file = glob.glob(directory + "/*.png")
    
    im = cv2.imread(img_file[0])
    im = cv2.resize(im, (512, 512))
    regions = []
    while True: 
        # Select ROI
        r = cv2.selectROI(im)

        # Crop image
        imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        with open(directory + '/regions.data','a') as f:
            f.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]) + "\n")
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
    '''