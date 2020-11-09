import sys
sys.path.append("./")

import nn
import update
import data

import numpy as np
import time

def forward(input, strides, k_shape):

        output = np.zeros((input.shape[0], input.shape[1], 
                                (input.shape[2] - k_shape[0]) // strides[0] + 1, 
                                (input.shape[3] - k_shape[1]) // strides[1] + 1))

        for n in np.arange(input.shape[0]):
            for c in np.arange(input.shape[1]):
                for i in np.arange(output.shape[2]):
                    for j in np.arange(output.shape[3]):
                        
                        output[n, c, i, j] = np.max(input[n, c,
                                                            strides[0]*i : strides[0]*i+k_shape[0],
                                                            strides[1]*j : strides[1]*j+k_shape[1]])
                        
        return output

def forward_fast(input, strides, k_shape):

        output = np.zeros((input.shape[0], input.shape[1], 
                                (input.shape[2] - k_shape[0]) // strides[0] + 1, 
                                (input.shape[3] - k_shape[1]) // strides[1] + 1))


        for i in np.arange(output.shape[2]):
            for j in np.arange(output.shape[3]):

                output[:, :, i:i+1, j:j+1] = input[:, :,
                                                    strides[0]*i : strides[0]*i+k_shape[0],
                                                    strides[1]*j : strides[1]*j+k_shape[1]].max(axis=2,keepdims=True).max(axis=3, keepdims =True)
                        
        return output

x = np.arange(1000000).reshape(100,10,100,10)
t = time.time()
forward(x,(2,2),(2,2))
print('100x10x100x10 original: ' + str(time.time()-t))
t = time.time()
forward_fast(x,(2,2),(2,2))
print('100x10x100x10 faster: ' + str(time.time()-t))