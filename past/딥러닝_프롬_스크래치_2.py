##########################################################
# 2.3장
import numpy as np

def AND(x1, x2) :
    x = np.array( [x1, x2] )
    w = np.array( [0.5, 0.5] )
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0 :
        #print("0")
        return 0
    elif tmp > 0 :
        #print("1")
        return 1


def NAND(x1, x2) :
    x = np.array( [x1, x2] )
    w = np.array( [0.5, 0.5] )
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp > 0 :
        #print("0")
        return 0
    elif tmp <= 0 :
        #print("1")
        return 1


def OR(x1, x2) :
    x = np.array( [x1, x2] )
    w = np.array( [0.5, 0.5] )
    b = -0.7
    bool_OR = False
    for i in x :    # 어느 것 하나라도 1이 있다면, OR 은 True
        if i == 1 :
            bool_OR = True
    if bool_OR == True :    # 어느 것 하나라도 1이 있다면, 모든 가중치를 더함
        tmp = np.sum(w) + b
    else :
        tmp = b
    if tmp <= 0 :
        #print("0")
        return 0
    elif tmp > 0 :
        #print("1")
        return 1
############################################################
# 2.5장 XOR 게이트 구현, 다층 퍼셉트론(multi-layer perceptron)
def XOR(x1, x2) :
    # 다층 퍼셉트론(multi-layer perceptron) : 여러개의 논리 회로로 이루어진 퍼셉트론
    # 자기자신 하나만 1이면 1 출력
    # x1, x2 : 0층
    x1_1 = NAND(x1, x2) # 1층
    x2_1 = OR(x1, x2)   # 1층
    return_value = AND(x1_1, x2_1)  # 2층
    return return_value


def main() :
    print("AND gate")
    print("AND(0, 0) : ", AND(0, 0))
    print("AND(0, 1) : ", AND(0, 1))
    print("AND(1, 0) : ", AND(1, 0))
    print("AND(1, 1) : ", AND(1, 1))
    print("NAND gate")
    print("NAND(0, 0) : ", NAND(0, 0))
    print("NAND(0, 1) : ", NAND(0, 1))
    print("NAND(1, 0) : ", NAND(1, 0))
    print("NAND(1, 1) : ", NAND(1, 1))
    print("OR gate")
    print("OR(0, 0) : ", OR(0, 0))
    print("OR(0, 1) : ", OR(0, 1))
    print("OR(1, 0) : ", OR(1, 0))
    print("OR(1, 1) : ", OR(1, 1))
    print("XOR gate")
    print("XOR(0, 0) : ", XOR(0, 0))
    print("XOR(0, 1) : ", XOR(0, 1))
    print("XOR(1, 0) : ", XOR(1, 0))
    print("XOR(1, 1) : ", XOR(1, 1))

main()
