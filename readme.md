## chapter 2
### 퍼셉트론
#### 다수의 입력값을 받아 0 또는 1을 출력하는 알고리즘
### 단층 논리회로
#### AND, NAND, OR
### 다층 논리회로
#### XOR = AND(OR, NAND)

### <br/><br/><br/>

## chapter 3
### step function (계단 함수)
#### b : bias라고 하며 b = -threshold 이다.
#### w : weight, x값에 대한 가중치이다.
```
tmp = b + w1*x1 + w2*x2
if tmp <= 0 : 
    return 0
else : 
    return 1
```
#### 특정 threshold를 넘으면 1을 출력하는 함수

### sigmoid (s자 모양)
* h(x) = 1 / (1 + exp(-x)) <br/> * exp는 자연상수 e = 2.7182...
#### x가 커질 수록 1로 수렴, 작아질 수록 0으로 수렴한다.
<div align="center"><img src="https://github.com/Shin-jongwhan/deep_learning_from_scratch/blob/master/ch3_sigmoid_and_setp_func.png" width="50%" height="50%"><br/></div>

### 선형 함수와 비선형 함수
* 선형 함수 : f(x) = ax + b의 형태로 나타낼 수 있는 함수 <br/>
선형 함수는 신경망에 쓸 수 없는 함수이다. 왜냐면 아무리 함수를 많이 써도 하나의 함수로 압축되기 때문이다. <br/>
예를 들어 func_1(x) = x + a, func_2(x) = x * b 함수가 있으면 func_2(func_1(x)) = (x + a) * b와 같이 두 함수는 하나로 써질 수 있기 때문이다.
* 비선형 함수 : 계단 함수, sigmoid 함수와 같이 직선 한 개로 그릴 수 없는 함수로, 다층을 구성할 수 있다.

### 행렬과 벡터(스칼라), 행렬의 곱
#### 행렬은 2차원 배열, 스칼라는 1차월 배열이다.
#### 행렬을 곱할 때는 A 행렬의 '열 개수', B 행렬의 '행 개수'가 같아야 한다.
#### 행렬 * 스칼라할 때에도 같은 규칙을 따른다. 스칼라는 1차원 배열로 열의 개수와 같다고 생각하면 된다.
```
# 행렬의 곱연산
# 아래 예시는 3개의 뉴런이 계산되는 것이다.
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
```
```
    # 입력층, 2개 은닉층, 출력층을 가진 신경망
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
```

### <br/><br/><br/>
