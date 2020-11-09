import sys
sys.path.append("./")

import nn
import numpy as np
import data
import update
import calculate

x = np.arange(144).reshape((3, 1, 6, 8))
x[:,:,:,2:-2] = 0
# k = np.array([[[[1.0, -1.0]]]])
y = np.arange(9).reshape((3,1,1,3))

updater = update.BGD(0.0001)

class Net():

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,2,(5,5))#2x2x4
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(2,1,(2,2))#1x2x3
        self.r2 = nn.ReLU()
    def forward(self, input):

        y = self.c1.forward(input)
        y = self.r1.forward(y)
        y = self.c2.forward(y)
        y = self.r2.forward(y)
        return y
    
    def backward(self, dldy):

        dldx = self.r2.backward(dldy)
        dldx = self.c2.backward_withUpdate(dldx, updater)
        dldx = self.r1.backward(dldx)
        dldx = self.c1.backward_withUpdate(dldx, updater)

        return dldx

net = Net()
loss_f = nn.MSError()

for i in range(500):

    y_hat = net.forward(x)
    dldy = loss_f.backward(y_hat, y)

    net.backward(dldy)
    print(i)

print(net.forward(x))

    