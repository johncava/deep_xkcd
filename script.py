import cv2
import numpy as np
import sys
import glob

if __name__ == '__main__' :
 
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

