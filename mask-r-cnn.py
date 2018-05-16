import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from utility import *

num_data = 20
data = read_region_data(get_list(20))

#v = Variable(torch.rand(1,1,512,512), requires_grad = False)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(1,64,kernel_size=3)
        self.conv2 = nn.Conv2d(64,64,kernel_size = 3)
        self.deconv = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(64,1,kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,image):
        #image = Variable(torch.Tensor(image).view(1,1,224,224), requires_grad = False)
        image = self.conv(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image, indices1 = self.pool(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        size1 = image.size()
        image, indices2 = self.pool(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        size2 = image.size()
        image, indices3 = self.pool(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        size3 = image.size()
        image, indices4 = self.pool(image)
        #size4 = image.size()

        image = self.unpool(image, indices4,output_size=size3)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.unpool(image, indices3,output_size=size2)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.unpool(image, indices2,output_size=size1)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.unpool(image, indices1)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv2(image)
        #image = self.relu(image)
        image = self.sigmoid(image)
        return image

max_epochs = 5
learning_rate = 1e-1
model = Model()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
loss_array = []

for epoch in xrange(max_epochs):
    for iteration in xrange(len(data)):
        img_data = data[iteration][0]
        regions_list = data[iteration][1]
        m = [[0]*512]*512
        m = np.array(m)
        for r in regions_list:
            m[(r[1]):(r[1]+r[3]),r[0]:(r[0]+r[2])] = 1
        x = Variable(torch.Tensor(img_data), requires_grad = False).view(1,1,512,512)
        y = Variable(torch.Tensor(m), requires_grad = False).view(1,1,512,512)
        prediction = model(x) 
        loss = loss_fn(prediction, y)
        print "Index ", iteration
        loss_array.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print "Done"
np.save("loss14.npy", loss_array)