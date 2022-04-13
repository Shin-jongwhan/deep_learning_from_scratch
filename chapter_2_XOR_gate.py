import numpy as np

def AND(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

def OR(x1, x2) :
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

def NAND(x1, x2) :      # AND와 부호 반대
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0 :
        return 0
    else :
        return 1


def XOR(x1, x2) :
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    output = AND(s1, s2)
    
    return output


def main() :
    output = XOR(0, 0)
    print("XOR(0, 0) : ", output)
    output = XOR(1, 0)
    print("XOR(1, 0) : ", output)
    output = XOR(0, 1)
    print("XOR(0, 1) : ", output)
    output = XOR(1, 1)
    print("XOR(1, 1) : ", output)


main()
