import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# 처음 한 번은 몇 분 정도 걸린다.
# normalize
## true : 이미지의 픽셀 값을 0 ~ 1.0으로 정규화시킴
## false : 이미지의 픽셀 값을 0 ~ 255의 값으로 그대로 출력
# flatten
## true : 1 * 28 * 28의 3차원 배열
## false : 784개 원소로 이뤄진 1차원 배열로 바꿈
# ont-hot-incoding
## true : 정답이 1이면 t_train[0] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]와 같이 정답을 뜻하는 원소만 1로 출력
## false : 정답인 숫자를 그대로 저장 ex) 정답이 7인 경우 t_train[0] = 7
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

# 각 데이터의 형상 출력
# 784 = 28 * 28
print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, )
print(x_test.shape)    # (10000, 784)
print(t_test.shape)    # (10000, )

#print(x_train[0][0])
print(t_train[0])
print(t_train[1])
print(t_train[2])
