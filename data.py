import numpy as np
import os
import gzip

def getBatch(dataSet, dataSet_value, batchSize=1):

    num_set = len(dataSet)
    indices = list(range(num_set))# 创建提取顺序列表

    np.random.shuffle(indices)# 打乱

    for i in range(0, num_set, batchSize):
        j = indices[i: min(i + batchSize, num_set)]# 截取顺序列表
        yield dataSet[j], dataSet_value[j]

#网上找的
def load_mnist():

    path='./dataset/FashionMNIST'
    kind='train'

    train_labels_path = os.path.join(path,

                               '%s-labels-idx1-ubyte.gz'

                               % kind)

    train_images_path = os.path.join(path,

                               '%s-images-idx3-ubyte.gz'

                               % kind)

    with gzip.open(train_labels_path, 'rb') as lbpath:

        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,

                               offset=8).reshape((-1, 1))

    with gzip.open(train_images_path, 'rb') as imgpath:

        train_images = np.frombuffer(imgpath.read(), dtype=np.uint8,

                               offset=16).reshape((len(train_labels), 1, 28, 28))

    kind='t10k'

    t10k_labels_path = os.path.join(path,

                               '%s-labels-idx1-ubyte.gz'

                               % kind)

    t10k_images_path = os.path.join(path,

                               '%s-images-idx3-ubyte.gz'

                               % kind)
    with gzip.open(t10k_labels_path, 'rb') as lbpath:

        t10k_labels = np.frombuffer(lbpath.read(), dtype=np.uint8,

                               offset=8).reshape((-1, 1))

    with gzip.open(t10k_images_path, 'rb') as imgpath:

        t10k_images = np.frombuffer(imgpath.read(), dtype=np.uint8,

                               offset=16).reshape((len(t10k_labels), 1, 28, 28))

    return train_images, train_labels, t10k_images, t10k_labels

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]

def labelsToArray(labels, num_kind):
    labels_array = np.zeros((labels.shape[0], num_kind))
    for n in range(labels.shape[0]):
        labels_array[n][labels[n][0]] = 1
    return labels_array
