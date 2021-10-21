### 3장 신경망
########################################
# 3.2.3 계단함수의 그래프
import numpy as np
import matplotlib.pylab as plt

def step_function(x) :
    # x array 에서 0 보다 큰 것을 True 로 지정, 나머진 false 로 지정. 그리고 dtype = np.int 로 하여 True == 1, False == 0으로 변환
    return np.array(x > 0, dtype = np.int)


def chap_3_2_3() : 
    x = np.arange(-5.0, 5.0, 0.1)   # -5.0 부터 5.0 까지 0.1 씩 더한 array 생성, 5.0 은 포함 X, 4.9 까지
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y 축 범위 지정
    plt.show()
########################################
# 3.2.4 시그모이드 함수 구현하기
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))


def chap_3_2_4() :
    x = np.array( [-1.0, 1.0, 2.0] )
    return_value = sigmoid(x)
    print(return_value)

    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    print(x)
    print(y)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def main() :
    chap_3_2_4()


main()
