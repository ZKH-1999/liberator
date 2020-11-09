import sys
sys.path.append("./")

import nn
import numpy as np
import update
import calculate
import data
import time

#读取数据集
train_images, train_labels, t10k_images, t10k_labels = data.load_mnist()

class Net():

    def __init__(self):#1x28x28

        self.c1 = nn.Conv2d(1, 6, (5,5))#6x24x24
        self.bn1 = nn.BatchNormalization(True, 6, 0.1)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d((2,2))#6x12x12

        self.c2 = nn.Conv2d(6, 16, (5,5))#16x8x8
        self.bn2 = nn.BatchNormalization(True, 16, 0.1)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d((2,2))#16x4x4

        self.l3 = nn.Linear(4*4*16, 120)
        self.bn3 = nn.BatchNormalization(False, 120, 0.1)
        self.r3 = nn.ReLU()

        self.l4 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNormalization(False, 84, 0.1)
        self.r4 = nn.ReLU()

        self.l5 = nn.Linear(84, 10)

    def forward(self, input):

        y = self.c1.forward(input)
        y = self.bn1.forward(y)
        y = self.r1.forward(y)
        y = self.p1.forward(y)

        y = self.c2.forward(y)
        y = self.bn2.forward(y)
        y = self.r2.forward(y)
        y = self.p2.forward(y)

        y = y.reshape((y.shape[0], 4*4*16))

        y = self.l3.forward(y)
        y = self.bn3.forward(y)
        y = self.r3.forward(y)

        y = self.l4.forward(y)
        y = self.bn4.forward(y)
        y = self.r4.forward(y)
        y = self.l5.forward(y)

        return y

    def backward(self, dldy, updater):

        dldx = self.l5.backward_withUpdate(dldy, updater)

        dldx = self.r4.backward(dldx)
        dldx = self.bn4.backward_withUpdate(dldx, updater)
        dldx = self.l4.backward_withUpdate(dldx, updater)

        dldx = self.r3.backward(dldx)
        dldx = self.bn3.backward_withUpdate(dldx, updater)
        dldx = self.l3.backward_withUpdate(dldx, updater)

        dldx = dldx.reshape((dldx.shape[0],16,4,4))

        dldx = self.p2.backward(dldx)
        dldx = self.r2.backward(dldx)
        dldx = self.bn2.backward_withUpdate(dldx, updater)
        dldx = self.c2.backward_withUpdate(dldx, updater)

        dldx = self.p1.backward(dldx)
        dldx = self.r1.backward(dldx)
        dldx = self.bn1.backward_withUpdate(dldx, updater)
        dldx = self.c1.backward_withUpdate(dldx, updater)

        return dldx

    def train(self):
        
        self.bn1.train()
        self.bn2.train()
        self.bn3.train()
        self.bn4.train()

    def eval(self):

        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
        self.bn4.eval()

net = Net()

loss_f = nn.CrossEntropyWithSoftMax()

epoch, lr = 5, 0.1

updater = update.BGD(lr)

def getTestAcc(myNet, testSet, testSet_value):
    myNet.eval()
    num_right, num_test = 0, 0
    for x, y in data.getBatch(testSet, testSet_value, 500):
        y_hat = myNet.forward(x)
        y_hat = calculate.Softmax(y_hat)
        for i in range(y.shape[0]):
            label = y_hat[i].argmax()
            if label == y[i]:
                num_right += 1
        num_test += y_hat.shape[0]
    return num_right/num_test
    
            
for i in range(epoch):
    with open('testdata.txt','a') as testdata:
        print('epoch '+str(i+1)+', training...')
        testdata.write('epoch '+str(i+1)+', training...\n')
        t = time.time()
        loss_t = 0
        n = 0
        batch = 1
        for x, y in data.getBatch(train_images, train_labels, 32):
            net.train()
            t_batch = time.time()
            loss_batch = 0
            y = data.labelsToArray(y, 10)
            y_hat = net.forward(x)
            loss_t += loss_f.forward(y_hat, y)
            loss_batch += loss_f.forward(y_hat, y)
            dldy = loss_f.backward(y_hat, y)
            net.backward(dldy, updater)
            n += y.shape[0]
            print('batch No.' + str(batch) + ", time cost : " + str(time.time()-t_batch) + ", loss is : "+ str(loss_batch.sum()/x.shape[0]))
            testdata.write('batch No.' + str(batch) + ", time cost : " + str(time.time()-t_batch) + ", loss is : "+ str(loss_batch.sum()/x.shape[0])+'\n')
            batch += 1
        print("epoch "+ str(i+1) + ", loss is : "+ str(loss_t.sum()/n)+ " time cost : " + str(time.time()-t))
        testdata.write("epoch "+ str(i+1) + ", loss is : "+ str(loss_t.sum()/n)+ " time cost : " + str(time.time()-t)+'\n')
        print("epoch "+ str(i+1) + ', testing...')
        testdata.write("epoch "+ str(i+1) + ', testing...\n')
        acc = getTestAcc(net, t10k_images, t10k_labels)
        print("epoch "+ str(i+1) + ', test acc : ' + str(acc))
        testdata.write("epoch "+ str(i+1) + ', test acc : ' + str(acc)+'\n')