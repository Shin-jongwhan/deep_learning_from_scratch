# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
    #return x_train, t_train


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def main() : 
    x, t = get_data()
    network = init_network()
    print("x.shape", x.shape)   # (10000, 784)      # 10000개 데이터셋
    print("x[0].shape", x[0].shape) # (784,) flatten으로 1 * 28 * 28을 1차원 배열로 바꿈
    print("network['W1'].shape", network['W1'].shape)   # (784, 50)     # 50개의 뉴런(임의의 값)
    print("network['W2'].shape", network['W2'].shape)   # (50, 100)     # 100개의 뉴런(임의의 값)
    print("network['W3'].shape", network['W3'].shape)   # (100, 10)     # 0~9 총 10개의 출력값

    batch_size = 100
    accuracy_cnt = 0
    
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        # axis = 1 : 각 첫 번째 차원에서 확률이 가장 높은 원소의 인덱스를 얻는다.
        p = np.argmax(y_batch, axis = 1)
        accuracy_cnt += np.sum(p == t[i : i + batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


main()

"""
# argmax(x, axis = 1)
# axis = 1 : 각 첫 번째 차원에서 확률이 가장 높은 원소의 인덱스를 얻는다.
>>> x = np.array( [[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]] )
>>> y = np.argmax(x, axis = 1)
>>> print(y)
[1 2 1 0]
>>> y[0]
1
"""

"""
# np.sum(y==t)
>>> y = np.array([1,2,1,0])
>>> t = np.array([1,2,0,0])
>>> print(y==t)
[ True  True False  True]
>>> np.sum(y==t)
3
"""
