import numpy as np

def Sigmoid(input):

    return 1/(1+np.exp(-input))

def Sigmoid_der(input):

    return Sigmoid(input)*(1-Sigmoid(input))

def ReLU(input):

    return (np.abs(input)+input) / 2

def ReLU_der(input):
    
    tem = np.zeros_like(input)
    biggerThanZero = input > tem
    return np.ones_like(input) * biggerThanZero

def Softmax(input):

    input_exp = np.exp(input)
    partition = input_exp.sum(axis = 1, keepdims=True)
    return input_exp/partition

def corr2d(input, k):
    output = np.zeros((input.shape[0]-k.shape[0]+1, input.shape[1]-k.shape[1]+1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = (input[i:i+k.shape[0], j:j+k.shape[1]]*k).sum()

    return output

def corr2d_MulIn(input, k):

    if(input.shape[0] != k.shape[0]):
        print(input.shape[0])
        print(k.shape[0])
        print("输入的通道数与卷积核层数不同")
        return
    output = corr2d(input[0], k[0])
    for i in range(1, input.shape[0]):
        output += corr2d(input[i], k[i])

    return output

def corr2d_MulOut_MulIn(input, k):
    output = []
    for i in range(0, k.shape[0]):
        output.append(corr2d_MulIn(input, k[i]))

    return np.array(output)

def flip180(arr):

    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def addZeros_arround(n, s, w, e, input):

    new_arr = np.zeros((input.shape[0]+n+s, input.shape[1]+w+e))
    new_arr[n:new_arr.shape[0]-s, w:new_arr.shape[1]-e] += input

    return new_arr