import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    im = cv2.imread("MapProjections.png")
    regions = []
    while True: 
        # Select ROI
        r = cv2.selectROI(im)

        # Crop image
        imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        with open('regions.data','a') as f:
            f.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]) + "," + str(r[3]))
        # Display cropped image
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
