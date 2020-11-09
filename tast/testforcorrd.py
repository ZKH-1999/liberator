import sys
sys.path.append("./")

import nn
import numpy as np
import data
import update
import calculate

x = np.ones((1, 1, 6, 8))
x[:,:,:,2:-2] = 0
k = np.array([[[[1.0, -1.0]]]])
y = np.array([calculate.corr2d_MulOut_MulIn(x[0],k)])


net = nn.Conv2d(1,1,(1,2))
loss_f = nn.MSError()
updater = update.BGD(0.001)

for i in range(2000):
    y_hat = net.forward(x)
    l = loss_f.forward(y_hat, y)

    dldy = loss_f.backward(y_hat,y)

    net.backward_withUpdate(dldy, updater)


print(net.forward(x))
    