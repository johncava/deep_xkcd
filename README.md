# deep_xkcd
Deep Learning of Comic Strips of XKCD to extract text

# Dataset
Images from XKCD, and transcript text scraped from ExplainXKCD

# Method
OpenCV was used to manually label regions of text in 18 images. Out of many trial and errors, the following is the loss of the best model trained on the images for 30 epochs. 

![alt text](https://raw.githubusercontent.com/johncava/deep_xkcd/master/result_loss33.png)

Phase 0: Preprocess Image into a 512 x 512 image and make it be grayscaled

![alt text](https://raw.githubusercontent.com/johncava/deep_xkcd/master/XKCD_Comic_982.png)

Phase 1: Use trained model to predict areas of text

![alt text](https://raw.githubusercontent.com/johncava/deep_xkcd/master/Prediction_982%5BThreshold%20%3E%200.5%5D.png)

Phase 2: Use iterative graph search in order to find the top left and bottom right bounding regions to extract

![alt text](https://raw.githubusercontent.com/johncava/deep_xkcd/master/Extracted_Text_Comic_982.png)

This can then be further used to then mask the text from the original comic image

![alt text](https://raw.githubusercontent.com/johncava/deep_xkcd/master/XKCD_Comic_982_Mask.png)

# Future/In Progress

1. Train a CNN-LSTM model to "caption" extracted text into a transcript.
2. Manual label characters as well, and find a way to relate characters to the text they said.
3. Have more labeled images (>100)
