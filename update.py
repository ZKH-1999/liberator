#本类负责对data和grad域进行操作运算
import numpy as np



class BGD():

    #为什么全可选？ 为了实现with方法
    def __init__(self, lr, parametersList=[], batchSize=1, weight_decay=0):

        self.parametersList = parametersList
        self.batchSize = batchSize
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self):

        for parameter in self.parametersList:
            # print(parameter.data)
            parameter.data -= self.lr*parameter.grad/self.batchSize + (self.weight_decay/parameter.data.size * self.lr * parameter.data)
            # print(parameter.data)

    def gradToZero(self):

        for parameter in self.parametersList:
            parameter.grad = np.zeros(parameter.data.shape)
    