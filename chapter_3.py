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


def main() :
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

    


main()



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
