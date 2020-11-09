import sys
sys.path.append("./")

import nn
import numpy as np
import data
import update


#使用线性回归测试 已通过
# num_inputs = 4
# num_examples = 10000

# truew_i_h = np.array([[10, 12, 14], 
#                     [-3.4, 1, 2], 
#                     [1, 1, 1], 
#                     [1, 1, 1]])
# truew_h_o = np.array([[3], [2], [1]])

# trueb_i_h = np.array([[4.2, 5, 6]])
# trueb_h_o = np.array([[1]])

# np.random.seed(2)
# examples = np.array(np.random.randn(num_inputs*num_examples)).reshape(num_examples, num_inputs)
# examples_trueValue = np.matmul(examples, truew_i_h) + trueb_i_h
# examples_trueValue = np.matmul(examples_trueValue, truew_h_o) + trueb_h_o
# examples_trueValue += np.random.normal(0, 0.01, examples_trueValue.shape)
# print(examples_trueValue)


# 非线性测试 已通过
examples = np.array([[0.05, 0.1]])
examples_trueValue = np.array([[0.01, 0.009]])

updater = update.BGD(0.1)

class Net():

    def __init__(self):
        
        self.l1 = nn.Linear(2,2)
        self.si1 = nn.Sigmoid()
        self.l2 = nn.Linear(2,2)
        self.si2 = nn.Sigmoid()

    def forward(self, input):

        y1 = self.l1.forward(input)
        y1 = self.si1.forward(y1)
        y1 = self.l2.forward(y1)
        output = self.si2.forward(y1)        
        
        return output

    def backward(self, dldy):
        dldx1 = self.si2.backward(dldy)
        dldx1 = self.l2.backward_withUpdate(dldx1, updater)
        dldx1 = self.si1.backward(dldx1)
        dldx2 = self.l1.backward_withUpdate(dldx1, updater)

net = Net()
loss_f = nn.MSError()

for i in range(100000):
    for x, y in data.getBatch(examples, examples_trueValue, 1): 

        y_hat = net.forward(x)
        dldy = loss_f.backward(y_hat, y)
        net.backward(dldy)

print(net.forward(examples))
    