#写各种层的类
#损失函数、激活函数也统一视为层
import numpy as np
import calculate



#定义参数的数据类型:



class parameter():
    
    def __init__(self, shape):
        self.shape = shape
        self.data = np.random.normal(0, 0.01, shape)
        self.grad = np.zeros(shape)



#layer:



class Linear():

    def __init__(self, num_input, num_output, hasBias=True):

        self.weight = parameter((num_input,num_output))
        if hasBias:
            self.bias = parameter((1,1))
        self.input = np.array([[0]])
    
    def forward(self, input):

        self.input = input
        output = np.matmul(input, self.weight.data) + self.bias.data

        return output
    
    #负责计算出grad，更新用update里的方法
    def backward(self, dldy):

        dldx = np.matmul(dldy, self.weight.data.T)

        self.weight.grad = np.matmul(self.input.T, dldy)
        self.bias.grad = dldy.sum()

        return dldx

    #写一些为什么加入updater参数，可以不将参数传来传去同时又实现了多种优化函数都可以用，也不用担心
    #简化流程的同时又节约了内存
    def backward_withUpdate(self, dldy, updater):
        
        dldx = self.backward(dldy)

        updater.parametersList = self.getParametersList()
        updater.batchSize = self.input.shape[0]

        updater.update()
        updater.gradToZero()

        return dldx

    def getParametersList(self):
    
        parametersList = [self.weight, self.bias]

        return parametersList

#无填充、步长固定为1
class Conv2d():
    
    def __init__(self, in_channel, out_channel, k_shape):
        
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        shape = (out_channel, in_channel, k_shape[0], k_shape[1])
        self.k = parameter(shape)
        self.bias = parameter((1,1))
        self.input = np.array([[0]])

    def forward(self, input_niwh):

        self.input = input_niwh
        output = []

        for i in range(0, input_niwh.shape[0]):

            output.append(calculate.corr2d_MulOut_MulIn(input_niwh[i], self.k.data))
        
        return np.array(output) + self.bias.data
    
    def backward(self,dldy):

        dldw_noiwh = []#放入oiwh
        for no_batch in range(0, dldy.shape[0]):

            dldw_oiwh = []#放入iwh 组成一个n的oiwh
            for no_outchannel in range(0, dldy.shape[1]):
                dldw_iwh = []#放入 wh 组成一个o的iwh
                for no_inchannel in range(0, self.input.shape[1]):
                    dldw_iwh.append(calculate.corr2d(self.input[no_batch][no_inchannel], dldy[no_batch][no_outchannel]))
                dldw_oiwh.append(dldw_iwh)
                
            dldw_noiwh.append(dldw_oiwh)
        self.k.grad = np.array(dldw_noiwh).sum(axis=0)
        self.bias.grad = dldy.sum()

        dldx_niwh = []
        for no_batch in range(0, dldy.shape[0]):

            dldx_oiwh = []
            for no_outchannel in range(0, dldy.shape[1]):
                dldy_afterAddZeros = calculate.addZeros_arround(self.k.data.shape[2]-1,self.k.data.shape[2]-1,self.k.data.shape[3]-1,self.k.data.shape[3]-1,dldy[no_batch][no_outchannel])
                dldx_iwh = []
                for no_inchannel in range(0, self.k.data.shape[1]):
                    dldx_iwh.append(calculate.corr2d(dldy_afterAddZeros, calculate.flip180(self.k.data[no_outchannel][no_inchannel])))
                dldx_oiwh.append(dldx_iwh)
            dldx_niwh.append(np.array(dldx_oiwh).sum(axis=0))

        return np.array(dldx_niwh)

    def backward_withUpdate(self, dldy, updater):
        
        dldx = self.backward(dldy)

        updater.parametersList = self.getParametersList()
        updater.batchSize = self.input.shape[0]

        updater.update()
        updater.gradToZero()

        return dldx


    def getParametersList(self):
        parametersList = [self.k, self.bias]
        return parametersList

class MaxPool2d():

    def __init__(self, k_shape, strides=(2, 2)):

        super().__init__()
        self.k_shape = k_shape
        self.strides = strides
        self.input = np.array([[0]])

    def forward(self, input, strides=(2, 2)):

        self.input = input
        output = np.zeros((input.shape[0], input.shape[1], 
                                (input.shape[2] - self.k_shape[0]) // self.strides[0] + 1, 
                                (input.shape[3] - self.k_shape[1]) // self.strides[1] + 1))

        for n in np.arange(input.shape[0]):
            for c in np.arange(input.shape[1]):
                for i in np.arange(output.shape[2]):
                    for j in np.arange(output.shape[3]):
                        
                        output[n, c, i, j] = np.max(input[n, c,
                                                            self.strides[0]*i : self.strides[0]*i+self.k_shape[0],
                                                            self.strides[1]*j : self.strides[1]*j+self.k_shape[1]])
                        
        return output

    def backward(self, dldy):

        dldx = np.zeros_like(self.input)

        
        for n in np.arange(self.input.shape[0]):
            for c in np.arange(self.input.shape[1]):
                for i in np.arange(dldy.shape[2]):
                    for j in np.arange(dldy.shape[3]): 

                        maxIndex_cell = np.unravel_index(np.argmax(self.input[n, c,
                                                            self.strides[0]*i : self.strides[0]*i+self.k_shape[0],
                                                            self.strides[1]*j : self.strides[1]*j+self.k_shape[1]]),
                                                            self.input[n, c,
                                                            self.strides[0]*i : self.strides[0]*i+self.k_shape[0],
                                                            self.strides[1]*j : self.strides[1]*j+self.k_shape[1]].shape)
                        maxIndex = (n, c, maxIndex_cell[0]+self.strides[0]*i, maxIndex_cell[1]+self.strides[1]*j)

                        dldx[maxIndex] = dldy[n, c, i, j]

        return dldx

class BatchNormalization():

    def __init__(self, isConv, num_output, moving_updateRate):

        super().__init__()
        self.isTraining = True
        self.isConv = isConv
        self.eps = 1e-5
        self.moving_updateRate = moving_updateRate
        self.input = np.array([[0]])
        self.input_hat = np.array([[0]])
        self.batch_mean = np.array([[0]])
        self.batch_var = np.array([[0]])

        if isConv:

            self.r = parameter((1, num_output,1,1))
            self.b = parameter((1, num_output,1,1))

            self.moving_Mean = np.zeros((1, num_output,1,1))
            self.moving_var = np.zeros((1, num_output,1,1))

        else:

            self.r = parameter((1, num_output))
            self.b = parameter((1, num_output))

            self.moving_Mean = np.zeros((1, num_output))
            self.moving_var = np.zeros((1, num_output))

    def forward(self, input):
        if not self.isTraining:
                
            input_hat = (input-self.moving_Mean)/np.sqrt(self.moving_var + self.eps)

        else:

            if self.isConv:

                batch_mean = input.mean(axis=0,keepdims = True).mean(axis=2,keepdims = True).mean(axis=3,keepdims = True)
                batch_var = ((input-batch_mean)**2).mean(axis=0, keepdims = True).mean(axis=2, keepdims=True).mean(axis=3, keepdims=True)

            else:

                batch_mean = input.mean(axis = 0, keepdims = True)
                batch_var = ((input-batch_mean)**2).mean(axis = 0, keepdims = True)
            
            input_hat = (input-batch_mean)/np.sqrt(batch_var + self.eps)

            self.input = input
            self.input_hat = input_hat
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            self.moving_Mean = (1-self.moving_updateRate)*self.moving_Mean + self.moving_updateRate*batch_mean
            self.moving_var = (1-self.moving_updateRate)*self.moving_var + self.moving_updateRate*batch_var

        output = input_hat*self.r.data + self.b.data

        return output

    def backward(self, dldy):
        
        if self.isConv:
            
            self.r.grad = (dldy * self.input_hat).sum(axis=0, keepdims=True).sum(axis=2, keepdims=True).sum(axis=3, keepdims=True)
            self.b.gard = dldy.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True).sum(axis=3, keepdims=True)

            dldx_hat = dldy * self.r.data
            dx_hatdx = 1/np.sqrt(self.batch_var + self.eps)

            dx_hatdbatch_var = -0.5 * (self.input-self.batch_mean) * ((self.batch_var+self.eps)**-1.5)
            dldbatch_var = (dldx_hat*dx_hatdbatch_var).sum(axis=0, keepdims=True).sum(axis=2, keepdims=True).sum(axis=3, keepdims=True)
            dbatch_vardx = 2*(self.input - self.batch_mean)/self.input.shape[0]

            dx_hatdbatch_mean = - dx_hatdx
            dbatch_vardbatch_mean = - dbatch_vardx
            dldbatch_mean = (dldx_hat*dx_hatdbatch_mean).sum(axis=0, keepdims=True).sum(axis=2, keepdims=True).sum(axis=3, keepdims=True) + dldbatch_var * (dbatch_vardbatch_mean.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True).sum(axis=3, keepdims=True))
            dbatch_meandx = 1/self.input.shape[0]

            dldx = dldx_hat * dx_hatdx + dldbatch_var * dbatch_vardx + dbatch_meandx * dldbatch_mean

        else:

            self.r.grad = (dldy * self.input_hat).sum(axis=0, keepdims=True)
            self.b.grad = dldy.sum(axis=0, keepdims=True)


            dldx_hat = dldy*self.r.data
            dx_hatdx = 1/np.sqrt(self.batch_var + self.eps)

            dx_hatdbatch_var = -0.5 * (self.input-self.batch_mean) * ((self.batch_var+self.eps)**-1.5)
            dldbatch_var = (dldx_hat*dx_hatdbatch_var).sum(axis=0,keepdims=True)
            dbatch_vardx = 2*(self.input - self.batch_mean)/self.input.shape[0]

            dx_hatdbatch_mean = - dx_hatdx
            dbatch_vardbatch_mean = - dbatch_vardx
            dldbatch_mean = (dldx_hat * dx_hatdbatch_mean).sum(axis = 0, keepdims = True) + dldbatch_var * (dbatch_vardbatch_mean.sum(axis=0, keepdims=True))
            dbatch_meandx = 1/self.input.shape[0]

            dldx = dldx_hat * dx_hatdx + dldbatch_var * dbatch_vardx + dbatch_meandx * dldbatch_mean

        return dldx

    def backward_withUpdate(self, dldy, updater):

        dldx = self.backward(dldy)

        updater.parametersList = self.getParametersList()
        updater.batchSize = self.input.shape[0]

        updater.update()
        updater.gradToZero()

        return dldx

    def train(self):
        
        self.isTraining = True

    def eval(self):

        self.isTraining = False

    def getParametersList(self):

        parametersList = [self.r, self.b]
        return parametersList        




#激活函数部分:



class Sigmoid():

    def __init__(self):

        super().__init__()
        self.input = np.array([[0]])
    
    def forward(self, input):

        self.input = input
        return calculate.Sigmoid(input)
    
    def backward(self, dldy):

        return dldy * calculate.Sigmoid_der(self.input)

class ReLU():

    def __init__(self):
        
        super().__init__()
        self.input = np.array([[0]])

    def forward(self, input):

        self.input = input
        return calculate.ReLU(input)
    
    def backward(self, dldy):

        return dldy * calculate.ReLU_der(self.input) 
        



#损失函数部分:



class MSError():

    def __init__(self):
        
        super().__init__()


    def forward(self, y_hat, trueValue):

        return ((y_hat-trueValue)**2).mean()

    def backward(self, y_hat, trueValue):

        return 2*(y_hat-trueValue)

class CrossEntropyWithSoftMax():

    def __init__(self):
        super().__init__()
        
    def forward(self, y_hat, trueValue):

        afterSoftmax = calculate.Softmax(y_hat)
        # print(afterSoftmax, trueValue)
        loss = -(trueValue*np.log(afterSoftmax)).sum(axis=1, keepdims=True)
        return loss

    def backward(self, y_hat, trueValue):

        afterSoftmax = calculate.Softmax(y_hat)
        return afterSoftmax-trueValue
