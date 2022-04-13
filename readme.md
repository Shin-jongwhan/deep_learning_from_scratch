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

### <br/><br/><br/>
