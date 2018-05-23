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


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(1,64,kernel_size=3)
        self.conv2 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv3 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv4 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv5 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv6 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv7 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv8 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv9 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv10 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv11 = nn.Conv2d(64,64,kernel_size = 3)
        self.conv12 = nn.Conv2d(64,64,kernel_size = 3)

        self.deconv = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv4 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv5 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv6 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv7 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv8 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv9 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv10 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv11 = nn.ConvTranspose2d(64,64,kernel_size=3)
        self.deconv12 = nn.ConvTranspose2d(64,1,kernel_size=3)

    def encode(self,image):
        #image = Variable(torch.Tensor(image).view(1,1,224,224), requires_grad = False)
        image = self.conv(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.conv3(image)
        image = self.relu(image)
        image, indices1 = self.pool(image)
        image = self.conv4(image)
        image = self.relu(image)
        image = self.conv5(image)
        image = self.relu(image)
        image = self.conv6(image)
        image = self.relu(image)
        size1 = image.size()
        image, indices2 = self.pool(image)
        image = self.conv7(image)
        image = self.relu(image)
        image = self.conv8(image)
        image = self.relu(image)
        image = self.conv9(image)
        image = self.relu(image)
        size2 = image.size()
        image, indices3 = self.pool(image)
        image = self.conv10(image)
        image = self.relu(image)
        image = self.conv11(image)
        image = self.relu(image)
        image = self.conv12(image)
        image = self.relu(image)
        size3 = image.size()
        image, indices4 = self.pool(image)
        #size4 = image.size()
        return image, indices1, indices2,indices3,indices4,size1,size2,size3

    def decode(self,image, indices1,indices2,indices3,indices4,size1,size2,size3):
        image = self.unpool(image, indices4,output_size=size3)
        image = self.deconv(image)
        image = self.relu(image)
        image = self.deconv2(image)
        image = self.relu(image)
        image = self.deconv3(image)
        image = self.relu(image)
        image = self.unpool(image, indices3,output_size=size2)
        image = self.deconv4(image)
        image = self.relu(image)
        image = self.deconv5(image)
        image = self.relu(image)
        image = self.deconv6(image)
        image = self.relu(image)
        image = self.unpool(image, indices2,output_size=size1)
        image = self.deconv7(image)
        image = self.relu(image)
        image = self.deconv8(image)
        image = self.relu(image)
        image = self.deconv9(image)
        image = self.relu(image)
        image = self.unpool(image, indices1)
        image = self.deconv10(image)
        image = self.relu(image)
        image = self.deconv11(image)
        image = self.relu(image)
        image = self.deconv12(image)
        image = self.relu(image)
        #image = self.sigmoid(image)
        return image

    def forward(self, image):
        image, i1,i2,i3,i4,s1,s2,s3 = self.encode(image)
        image = self.decode(image, i1,i2,i3,i4,s1,s2,s3)
        return image


max_epochs = 12
learning_rate = 1e-3
#model = Model()
model = autoencoder()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
loss_array = []

for epoch in xrange(max_epochs):
    for iteration in xrange(len(data)):
        img_data = data[iteration][0]
        img_data = np.array(img_data)
        img_data /= 255
        img_data = img_data.tolist()
        x = Variable(torch.Tensor(img_data).view(1,1,512,512), requires_grad = False)
        prediction = model(x) 
        loss = loss_fn(prediction, x)
        print "Index ", iteration
        loss_array.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print "Done"
np.save("autoencoder_loss5.npy", loss_array)
torch.save(model.state_dict(), "autoencoder.model")