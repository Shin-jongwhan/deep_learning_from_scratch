import sys, os
import numpy as np
from dataset.mnist import load_mnist        # dataset : 실행하는 스크립트 위치의 폴더 이름, mnist : .py 파일 이름
import pickle

sys.path.append(os.pardir)

def sigmoid(x) :
    return x / (1 + np.exp(-x))


def softmax(a) :
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data() :
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, t_test


def init_network() :
    with open("sample_weight.pkl", 'rb') as f :
        network = pickle.load(f)
    print(network.keys())
    lsKey_network = list(network.keys())

    for i in range(0, len(lsKey_network)) :
        #print(network[lsKey_network[i]])
        #print(type(network[lsKey_network[i]]))
        print(lsKey_network[i])
        print(network[lsKey_network[i]].shape)

    return network


def predict(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    #print(y)       # 10개 배열로 이루어진 np.array
    #print(np.sum(y))       # 1이 출력되야 한다.

    return y


def main() :
    x, t = get_data()
    # t : 이미 계산된 y 값이고, predict(network, x) 로 x 에서 y 값이 나오면, t 값과 비교해서 t 값과 똑같으면 예상한 값이 맞다고 accuracy_cnt += 1 해준다.
    # t 는 10000개의 numpy arrary 로 형상은 (10000, ), 0 ~ 9까지의 숫자로 되어 있다.
    print("x shape")
    print(x.shape)
    print("t shape")
    print(t.shape)
    #for i in range(0, len(t)) :
    #    print(t[i])
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)) :
        y = predict(network, x[i])
        p = np.argmax(y)        # 확률이 가장 높은 원소의 인덱스를 얻는다.
        if p == t[i] :
            accuracy_cnt += 1

    print("Accuracy : " + str(float(accuracy_cnt) / len(x)))

main()


#>>> 
#= RESTART: C:\Users\user\Desktop\python\deeplearning_from_scratch\딥러닝_프롬_스크래치_3_6_2.py
#Accuracy : 0.8321
