## chapter 2
### 퍼셉트론
#### 다수의 입력값을 받아 0 또는 1을 출력하는 알고리즘
### 단층 논리회로
#### AND, NAND, OR
### 다층 논리회로
#### XOR

## chapter 3
### step function (계단 함수)
```
tmp = b + w1*x1 + w2*x2
if tmp <= 0 : 
  return 0
else : 
  return 1
```
#### 특정 threshold를 넘으면 1을 출력하는 함수
### sigmoid (s자 모양)
#### h(x) = 1 / (1 + exp(-x)), exp는 자연상수 e = 2.7182...
#### x가 커질 수록 1로 수렴, 작아질 수록 0으로 수렴한다.
#### 
<div align="center"><img src="https://github.com/Shin-jongwhan/deep_learning_from_scratch/blob/master/ch3_sigmoid_and_setp_func.png" width="50%" height="50%"><br/></div>
