import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter
from utility import *

#v = Variable(torch.rand(1,1,224,224), requires_grad = False)
num_data = 20
data, corpus = read_data(get_list(20))
dictionary = Counter(corpus)
hot = create_hot(dictionary.keys())

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(1,64, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(64,64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64,128, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(128,128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128,256, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(256,256, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(256,256, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256,512, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(512,512, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(512,512, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.linear = nn.Sequential(
                nn.Linear(178,178),
                nn.ReLU(),
                nn.Linear(178,178),
                nn.ReLU(),
                nn.Linear(178,57),
                nn.ReLU()
                )
        self.lstm = nn.LSTM(57,57)
        self.hidden = self.init_hidden()
        self.Wa = Variable(torch.rand(64,1))
        self.Wh = Variable(torch.rand(57,1))

    def init_hidden(self):
        return (Variable(torch.zeros(1,1,57)),
                Variable(torch.zeros(1,1,57)))
    
    def calc_alpha_weights(self, a_list):
        alpha_weights = []
        sigma = torch.exp(F.tanh(torch.dot(a_list[0].view(-1), self.Wa.view(-1)) 
                + torch.mm(self.hidden[0].view(1,57),self.Wh)
                + torch.mm(self.hidden[1].view(1,57),self.Wh)))
        alpha_weights.append(sigma)
        for index in xrange(1,len(a_list)):
            a = torch.exp(F.tanh(torch.dot(a_list[index].view(-1), self.Wa.view(-1)) 
                + torch.mm(self.hidden[0].view(1,57),self.Wh)
                + torch.mm(self.hidden[1].view(1,57),self.Wh)))
            sigma = sigma + a
            alpha_weights.append(a)
        for index, aw in enumerate(alpha_weights):
            alpha_weights[index] = aw/sigma
        return alpha_weights

    def forward(self,image,input_):
        image = Variable(torch.Tensor(image).view(1,1,224,224), requires_grad = False)
        image = self.cnn(image)
        a_list = image.view(512,64)
        prediction_list = []
        for character in input_:
            out, self.hidden = self.lstm(character.view(1,1,-1),self.hidden)
            alpha = self.calc_alpha_weights(a_list)
            z = Variable(torch.zeros(1,64))
            for a,b in zip(a_list, alpha):
                z = z + a*b
            z_t = torch.cat((z, self.hidden[0].view(1,57),self.hidden[1].view(1,57)),1)
            prediction = self.linear(z_t)
            prediction_list.append(prediction)
        return prediction_list


max_epoch = 2
learning_rate = 1e-3
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
loss_array = []
for epoch in xrange(max_epoch):
    for index in xrange(len(data)):
        comic = data[index]
        image, transcript = comic[0], comic[1]
        t = [Variable(torch.Tensor(hot[x.lower()]), requires_grad = False) for x in transcript]
        input_ = [Variable(torch.Tensor(hot['<SOS>']),requires_grad = False)] + t
        output_ = [Variable(torch.Tensor(hot[y.lower()]), requires_grad = False).view(1,57).long() for y in transcript] + [Variable(torch.Tensor(hot['<EOS>']), requires_grad = False).view(1,57).long()]
        y_pred = model(image, input_)
        loss = 0
        model.hidden = model.init_hidden()
        for pred, expected in zip(y_pred,output_):
            loss = loss + loss_fn(pred.view(1,57), torch.max(expected,1)[1])
        print index #,loss.item()
        loss_array.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print "Done"
np.save("loss11.npy", loss_array)
