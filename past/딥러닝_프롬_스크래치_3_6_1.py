import sys, os
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def practice_3_6_1_1() :
    sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
    # 처음 한 번은 몇분 정도 걸립니다.
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

    # 각 데이터의 형상 출력
    print(x_train.shape)    # (60000, 784)
    print(t_train.shape)    # (60000, )
    print(x_test.shape) # (10000, 784)
    print(t_test.shape) # (10000, )
    
###############################################################

def img_show(img) :
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def practice_3_6_1_2() :
    sys.path.append(os.pardir)
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)
    img = x_train[0]
    print(img)
    label = t_train[0]
    print(label)        # 5

    print(img.shape)        # (784, )
    img = img.reshape(28, 28)       # 원래 이미지의 모양으로 변형
    print(img.shape)        # (28, 28)

    img_show(img)


def main() :
    practice_3_6_1_1()
    practice_3_6_1_2()


main()
