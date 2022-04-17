# deeplearning from scratch - chapter 3
import numpy as np
import matplotlib.pylab as plt

#def step_function(x) :      # 실수형만 받을 수 있는 함수(넘파이 배열 안 됨)
#    if x > 0 :
#        return 1
#    else :
#        return 0


def step_function(x) :
    # numpy의 각 배열의 항이 조건에 충족하면 true, 아니면 false
    # ex)
    # array([-1.0, 1.0, 2.0]) -> array([False, True, True])
    y = x > 0
    return y.astype(np.int)     # bool -> int형으로 변환


def sigmoid(x) :
    return 1 / (1 + np.exp(-x))


def relu(x) :       # rectified linear unit
    return np.maximum(0, x)


def init_network() :
    # 입력층, 2개 은닉층, 출력층을 가진 신경망
    network = {}
    network['W1'] = np.array( [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]] )
    network['b1'] = np.array( [0.1, 0.2, 0.3] )
    network['W2'] = np.array( [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]] )
    network['b2'] = np.array( [0.1, 0.2] )
    network['W3'] = np.array( [[0.1, 0.3], [0.2, 0.4]] )
    network['b3'] = np.array( [0.1, 0.2] )

    return network


def identity_func(x) :
    return x


def forward(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_func(a3)

    return y


def softmax(a) :
    # 이 함수는 단조증가 함수(a >= b일 때, f(a) >= f(b)가 성립하는 함수)이기 때문에
    # 현업에서는 그냥 생략해도 무관하다고 한다.
    # (어차피 제일 높은 값은 a >= b라면 a 이기 때문)
    c = np.max(a)       # 오버플로 방지
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def main() :
    """
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y, linestyle = "--", label = "step_func")
    plt.ylim(-0.1, 1.1)     # y축 범위
    #plt.savefig("./ch3_step_function.png")
    #plt.show()
    #plt.clf()       # reset plt

    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y, label = "sigmoid")
    plt.ylim(-0.1, 1.1)
    plt.title("sigmoid, step function")
    plt.legend()
    plt.savefig("./ch3_sigmoid_and_setp_func.png")
    #plt.show()
    """

    # 입력층, 2개 은닉층, 출력층을 가진 신경망
    """
    # 첫 input 2개 값, 첫 번째 층, 3개 뉴런
    X = np.array( [1.0, 0.5] )
    W1 = np.array( [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]] )
    B1 = np.array( [0.1, 0.2, 0.3] )
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(Z1)

    # 두 번째 층, 2개 뉴런
    W2 = np.array( [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]] )
    B2 = np.array( [0.1, 0.2] )
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)

    # 출력층, 2개 뉴런
    W3 = np.array( [[0.1, 0.3], [0.2, 0.4]] )
    B3 = np.array( [0.1, 0.2] )
    Y = np.dot(Z2, W3) + B3
    print(Y)
    """

    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)    # [0.31682708 0.69627909]

    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))        # 항상 1이다

    


main()


# 행렬의 차원 구하기 및 활용
"""
>>> a = np.array([1,2,3,4])
>>> print(a)
[1 2 3 4]
>>> np.ndim(a) 배열의 차원 수 인트형으로 반환
1
>>> a.shape 튜플형으로 배열의 차원 개수, 해당 차원의 배열 수를 반환
(4,)
>>> a.shape[0]
4
>>> a.shape[1]
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    a.shape[1]
IndexError: tuple index out of range
>>> a.shape[np.ndim(a) - 1]
4
"""

# 행렬의 곱연산
# 아래 예시는 3개의 뉴런이 계산되는 것이다.
"""
>>> X = np.array( [1.0, 0.5] )
>>> W = np.array( [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]] )
>>> B = np.array( [0.1, 0.2, 0.3] )
>>> print(X.shape)
(2,)
>>> print(W.shape)
(2, 3)
>>> print(B.shape)
(3,)
>>> A1 = np.dot(X, W) + B
>>> print(A1)
[0.3 0.7 1.1]    # 뉴런이 3개
"""
