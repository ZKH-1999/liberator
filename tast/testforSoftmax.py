import sys
sys.path.append("./")

import nn
import numpy as np
import data
import update

#读取数据集
examples = []
examples_truvalue = []

with open("./dataset/balance.dat", encoding="UTF-8") as dataSet:
    line = ''
    for line in dataSet:
        datas = line.split(',')
        example = [float(datas[0]), float(datas[1]), float(datas[2]), float(datas[3])]
        examples.append(example)
        if(datas[4] == ' L\n'):
            example_truvalue = [1, 0, 0]
        if(datas[4] == ' B\n'):
            example_truvalue = [0, 1, 0]
        if(datas[4] == ' R\n'):
            example_truvalue = [0, 0, 1]
        examples_truvalue.append(example_truvalue)

examples = np.array(examples)
examples_truvalue = np.array(examples_truvalue)

class Net():
    def __init__(self):
        self.l1 = nn.Linear(4,3)

    def forward(self, input):
        return self.l1.forward(input)
    
    def backward(self, dldy, updater):
        self.l1.backward_withUpdate(dldy, updater)

updater = update.BGD(0.01)
loss_f = nn.CrossEntropyWithSoftMax()
net = Net()

#预测函数
def predict(input):
    output = net.forward(input).reshape((-1, 1))
    if np.argmax(output) == 0:
        return " L\n"
    if np.argmax(output) == 1:
        return " B\n"
    if np.argmax(output) == 2:
        return " R\n"

#使用模型预测
num_true = 0
with open("./dataset/balance.dat", encoding="UTF-8") as dataSet:
    line = ''
    for line in dataSet:
        datas = line.split(',')
        example = [float(datas[0]), float(datas[1]), float(datas[2]), float(datas[3])]
        y_hat =  predict(np.array([example]))
        if y_hat == datas[4]:
            num_true += 1

print(num_true/625)

for i in range(50):
    for x,y in data.getBatch(examples, examples_truvalue, 50):
        y_hat = net.forward(x)

        dldy = loss_f.backward(y_hat, y)
        net.backward(dldy, updater)

#使用模型预测
num_true = 0
with open("./dataset/balance.dat", encoding="UTF-8") as dataSet:
    line = ''
    for line in dataSet:
        datas = line.split(',')
        example = [float(datas[0]), float(datas[1]), float(datas[2]), float(datas[3])]
        y_hat =  predict(np.array([example]))
        if y_hat == datas[4]:
            num_true += 1

print(num_true/625)
