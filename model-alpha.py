import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from collections import Counter
from utility import *

num_data = 20
data, corpus = read_data(get_list(20))
dictionary = Counter(corpus)
dictionary = create_hot(dictionary.keys())

input_ = Variable(torch.Tensor(data[0][0]).view(1,1,224,224), requires_grad = False)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(1,16, kernel_size=12,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,16, kernel_size=10,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16,15,kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(15, 15, kernel_size=3, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
    def forward(self,x):
        x = self.cnn(x)
        print x.size()
        x = x.view(15,169)
        return x

model = Model()

out = model(input_)
for i in out:
    print i
