import cv2
import numpy as np
import sys
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from utility import *

dataset = []
corpus = []
comics = []

with open('./dataset.txt') as f:
    for line in f:
        line = line.split('\n')
        comics.append(line[0])
comics = comics[-10:]
for comic in comics:
    directory = './xkcd_archive/' + str(comic)
    print comic
    regions = glob.glob(directory + "/regions.data")
    # Count the number of regions
    count = 1
    with open(directory + '/regions.data','r') as R:
        for line in R:
            count = count + 1
    for index in xrange(1, count):
        img = Image.open(directory + "/" + str(index) + ".png")
        x,y = img.size
        img_data = list(img.getdata(band = 0))
        img_data = np.array(img_data)
        img_data = np.reshape(img_data, (x,y))
        # Impose img_data into 512 x 512 canvas
        canvas = np.zeros((512,512))
        canvas[0:x ,0:y] = img_data[:]

        # Read in the text corresponding to the image
        character_list = []
        with open(directory + "/" + str(index) + ".txt") as T:
            text = T.read()
            text = text.strip("\n")
            for character in text:
                character_list.append(character)
        
        corpus = corpus + character_list
        # Append image and text label into dataset
        dataset.append([canvas, character_list])

print "Dataset Created ", len(dataset)
dictionary = Counter(corpus)
hot = create_hot(dictionary.keys())
print "Hot Encoding Created"
#print len(hot.keys()) KEYS == 56

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(4, 4, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_bn1 = nn.BatchNorm2d(1)
        self.conv_bn32 = nn.BatchNorm2d(32)
        self.conv_bn64 = nn.BatchNorm2d(64)

        self.conv = nn.Conv2d(1,32,kernel_size=12)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 8)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5)
        self.conv4 = nn.Conv2d(32,32,kernel_size = 5)
        self.conv5 = nn.Conv2d(32,64,kernel_size = 3)
        self.conv6 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv7 = nn.Conv2d(64,64,kernel_size = 3)

        self.deconv8 = nn.ConvTranspose2d(64,32,kernel_size=3)
        self.deconv9 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv10 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv11 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv12 = nn.ConvTranspose2d(32,1,kernel_size=5)

        self.lstm = nn.LSTM(56,2304)
        self.hidden = self.init_hidden()
        self.linear = nn.Sequential(
                nn.Linear(2304,56),
                nn.ReLU()
                )

    def init_hidden(self):
        return (Variable(torch.zeros(1,1,2304)),
                Variable(torch.zeros(1,1,2304)))

    def encode(self,image):
        #image = Variable(torch.Tensor(image).view(1,1,224,224), requires_grad = False)
        image = self.conv(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.conv3(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image, indices1 = self.pool(image)
        image = self.conv4(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.conv5(image)
        image = self.conv_bn64(image)
        image = self.relu(image)
        image, indices2 = self.pool(image)
        image = self.conv6(image)
        image = self.conv_bn64(image)
        image = self.relu(image)
        image = self.conv7(image)
        image = self.conv_bn64(image)
        image = self.relu(image)
        image, indices3 = self.pool(image)
        #size4 = image.size()
        return image, indices1, indices2, indices3

    '''
    def decode(self,image, indices1):
        image = self.deconv8(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.deconv9(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.unpool(image, indices1)
        image = self.deconv10(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.deconv11(image)
        image = self.conv_bn32(image)
        image = self.relu(image)
        image = self.deconv12(image)
        image = self.conv_bn1(image)
        image = self.relu(image)
        #image = self.sigmoid(image)
        return image
    '''
    def forward(self, image, input_):
        image = Variable(torch.Tensor(image).view(1,1,512,512), requires_grad = False)
        image, _, _, _ = self.encode(image)
        #image = self.decode(image, i1)
        prediction_list = []
        self.hidden = self.init_hidden()
        for character in input_:
            out, self.hidden = self.lstm(character.view(1,1,-1),self.hidden)
            prediction = self.linear(out.view(1,2304))
            prediction_list.append(prediction)
        return prediction_list

'''
x,y = dataset[-1]
model = Model()
x = Variable(torch.Tensor(x).view(1,1,512,512), requires_grad = False)
out = model(x)
print out.size()
'''

max_epoch = 6
learning_rate = 1e-4
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
loss_array = []
for epoch in xrange(max_epoch):
    for index in xrange(len(dataset)):
        #print "Enter"
        comic = dataset[index]
        image, transcript = comic[0], comic[1]
        t = [Variable(torch.Tensor(hot[x.lower()]), requires_grad = False) for x in transcript]
        input_ = [Variable(torch.Tensor(hot['<SOS>']),requires_grad = False)] + t
        output_ = [Variable(torch.Tensor(hot[y.lower()]), requires_grad = False).view(1,56).long() for y in transcript] + [Variable(torch.Tensor(hot['<EOS>']), requires_grad = False).view(1,56).long()]
        y_pred = model(image, input_)
        loss = 0
        for pred, expected in zip(y_pred,output_):
            loss = loss + loss_fn(pred.view(1,56), torch.max(expected,1)[1])
        print index #,loss.item()
        loss_array.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print "End"
print "Done"
np.save("loss34.npy", loss_array)
torch.save(model.state_dict(), "omega.model")
