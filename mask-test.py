import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from utility import *
from PIL import Image

num_data = 20
data = read_region_data(get_list(20))

#v = Variable(torch.rand(1,1,512,512), requires_grad = False)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.relu = nn.LeakyReLU()
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
        image = self.relu(image)
        #image = self.sigmoid(image)
        return image

model = Model()
model.load_state_dict(torch.load('mask-r-cnn.model'))

img_data = data[0][0]
regions_list = data[0][1]
#m = [[0]*512]*512
m = img_data
m = np.array(m)
x = Variable(torch.Tensor(img_data).view(1,1,512,512))
prediction = model(x).view(1,1,512,512).detach()
prediction = prediction.numpy()

# reshape to 2d
mat = np.reshape(prediction,(512,512))
# Creates PIL image
img = Image.fromarray(np.uint8(mat * 255) , 'L')
img.save('my.png')
img.show()
