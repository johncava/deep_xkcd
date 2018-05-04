import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

input_ = Variable(torch.Tensor(torch.rand(1,1,224,224)), requires_grad = False)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16, kernel_size=12,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,16, kernel_size=10,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(16,15,kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(15, 5, kernel_size=3, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
    def forward(self,x):
        x = self.layer1(x)
        print x.size()
        x = self.layer2(x)
        print x.size()
        x = self.layer3(x)
        print x.size()
        x = self.layer4(x)
        print x.size()
        return x

model = Model()

out = model(input_)
print out.view(1,845)
