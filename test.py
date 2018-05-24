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
        self.sigmoid = nn.Sigmoid()
        self.conv_bn1 = nn.BatchNorm2d(1)
        self.conv_bn32 = nn.BatchNorm2d(32)
        self.conv_bn64 = nn.BatchNorm2d(64)

        self.conv = nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 5)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5)
        self.conv4 = nn.Conv2d(32,32,kernel_size = 5)
        self.conv5 = nn.Conv2d(32,64,kernel_size = 3)

        self.deconv8 = nn.ConvTranspose2d(64,32,kernel_size=3)
        self.deconv9 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv10 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv11 = nn.ConvTranspose2d(32,32,kernel_size=5)
        self.deconv12 = nn.ConvTranspose2d(32,1,kernel_size=5)

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
        #size4 = image.size()
        return image, indices1

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

    def forward(self, image):
        image, i1 = self.encode(image)
        image = self.decode(image, i1)
        return image

model = Model()
model.load_state_dict(torch.load('mask-r-cnn-4[1e-3][epoch=30].model'))

img_data = data[2][0]
img_data = np.array(img_data)
np_img = img_data[:]
img_data /= 255
im = Image.fromarray(np.uint8(img_data * 255) , 'L')
img_data = img_data.tolist()
x = Variable(torch.Tensor(img_data).view(1,1,512,512), requires_grad = False)
prediction = model(x)

prediction = prediction.view(512,512)
prediction = prediction.data.numpy()


for i in xrange(512):
    for j in xrange(512):
        if prediction[i,j] > 0.5:
            prediction[i,j] = 1.0
        else:
            prediction[i,j] = 0.0

img = Image.fromarray(np.uint8(prediction * 255) , 'L')
img.show()
im.show()

grid = prediction.tolist()

def getIslands():
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    islands = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1.0:
                # count += 1
                stack = [(i,j)]
                connect = []
                for ii,jj in stack:
                    if 0<=ii<m and 0<=jj<n and grid[ii][jj] == 1.0:
                        connect.append((ii,jj))
                        grid[ii][jj] = 0.0
                        stack.extend([(ii+1,jj),(ii-1,jj),(ii,jj-1),(ii,jj+1)])
                islands.append(connect)
    return islands

islands = getIslands()
final_candidate_list = []
island_threshold = 100
for island in islands:
    if len(island) > island_threshold:
        final_candidate_list.append(island)

candidate_regions = []
for component in final_candidate_list:
    low_x, high_x, low_y, high_y = 512, 0, 512, 0
    for pixel in component:
        i,j = pixel
        if i < low_x:
            low_x = i
        if i > high_x:
            high_x = i
        if j < low_y:
            low_y = j
        if j > high_y:
            high_y = j
    candidate_regions.append((low_x, high_x, low_y, high_y))

for i in candidate_regions:
    low_x, high_x, low_y, high_y = i
    box = np_img[low_x:high_x, low_y:high_y]
    bbox = Image.fromarray(np.uint8(box * 255) , 'L')
    bbox.show()
    np_img[low_x:high_x, low_y:high_y] = 1.0

mask = Image.fromarray(np.uint8(np_img * 255) , 'L')
mask.show()
