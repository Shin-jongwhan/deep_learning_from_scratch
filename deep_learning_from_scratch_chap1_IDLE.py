#################################### 
### numpy
>>> # array 원소 수는 x, y 둘 다 같아야 하고 아니면 오류난다.
>>> import numpy as np
>>> x =  np.array( [1.0, 2.0, 3.0] )
>>> x	# 1차원 배열(벡터)
array([1., 2., 3.])
>>> y
[2.0, 4.0, 6.0]
>>> import numpy as np
>>> x =  np.array( [1.0, 2.0, 3.0] )
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
>>> y = ( [2.0, 4.0, 6.0] )
>>> x + y
array([3., 6., 9.])
>>> x * y
array([ 2.,  8., 18.])
>>> x / y
array([0.5, 0.5, 0.5])

>>> # numpy.array 는 numpy 배열의 스칼라값(array scalar)과는 바로 연산을 할 수 없고 형변환(int)을 해줘야 한다. 배열과 스칼라값의 산술연산을 브로드캐스트라고 한다.
>>> y * 1
[2.0, 4.0, 6.0]
>>> y * x[0]
Traceback (most recent call last):
  File "<pyshell#38>", line 1, in <module>
    y * x[0]
TypeError: can't multiply sequence by non-int of type 'numpy.float64'
>>> type(x[0])
<class 'numpy.float64'>
>>> y * int(x[0])
[2.0, 4.0, 6.0]

# 2차원 배열
A = np.array([ [1,2], [3, 4], [5, 6] ] )
>>> print(A)
[[1 2]
 [3 4]
 [5 6]]
>>> # A.shape : 행렬의 형상(몇 행, 몇 열)
>>> A.shape
(3, 2)
>>> A[0][0]
1
>>> # A.dtype : 원소의 타입
>>> A.dtype
dtype('int32')
>>> # 행렬의 수가 안 맞으면 행렬 수 표시가 되지 않는다.
>>> A = np.array([ [1,2], [3, 4], [5, 6, 7] ] )
>>> A.shape
(3,)

>>> # 벡터와 브로드캐스트하려면 행의 수가 같거나 열의 수가 같아야 하고(둘 중 하나가 같거나 둘 다 같아야 한다) 아니면 오류가 난다.
>>> A = np.array([ [1,2], [3, 4], [5, 6] ] )
>>> B = np.array( [10, 20] )
>>> C = np.array( [10, 20, 30] )
>>> D = ( [2], [3], [4] )
>>> A * B
array([[ 10,  40],
       [ 30,  80],
       [ 50, 120]])
>>> A * C
Traceback (most recent call last):
  File "<pyshell#73>", line 1, in <module>
    A * C
ValueError: operands could not be broadcast together with shapes (3,2) (3,) 
>>> A * D
array([[ 2,  4],
       [ 9, 12],
       [20, 24]])

>>> # 원소 접근(인덱스 접근)
>>> print(x)
[1. 2. 3.]
>>> print(A)
[[1 2]
 [3 4]
 [5 6]]
>>> A[0]
array([1, 2])
>>> A[0][1]
2
>>> for row in A :
	print(row)

>>> # 인덱스 접근 방법과 평탄화(flatten)
>>> A_1 = A.flatten()		 # A를 1차원 배열로 변환(평탄화)
>>> print(A_1)
[1 2 3 4 5 6]
>>> type(A_1)
<class 'numpy.ndarray'>
>>> A_1[0]
1
>>> A_1[1]
2
>>> A_1[-1]
6
>>> len(A_1)
6
>>> A_1[np.array([0, 2, 4])]	# 인덱스 0, 2, 4번 원소 값 얻기
array([1, 3, 5])
>>> A_1 > 2
array([False, False,  True,  True,  True,  True])
>>> type(A_1 > 2)	# type 은 여전히 numpy.ndarray
<class 'numpy.ndarray'>
>>> type( (A_1 > 2)[0]) 	# 각 원소의 type은 bool 형이다.
<class 'numpy.bool_'>
>>> (A_1 > 2)[0].dtype		# numpy.array 형의 type 확인
dtype('bool')
>>> type( (A_1 > 2)[0].dtype ) 		# (A_1 > 2)[0].dtype 자체의 type 이 bool 형은 아니다.
<class 'numpy.dtype'>
>>> A_1[A_1 > 2]
array([3, 4, 5, 6])
### numpy end
#####################################################
#####################################################
### matplotlib, 데이터의 시각화
# 그래프를 그려주는 library 이다.
>>> import matplotlib.pyplot as plt
>>> # x 에 0부터 6 에 도달하기 전까지 0.1을 더한 벡터를 생성한다.
>>> x = np.arange(0 , 6 , 0.1)
>>> # x 값에 sin 함수를 계산하여 y 변수에 할당한다.
>>> y = np.sin(x)
>>> print(x)
[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7
 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5
 3.6 3.7 3.8 3.9 4.  4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.  5.1 5.2 5.3
 5.4 5.5 5.6 5.7 5.8 5.9]
>>> print(y)
[ 0.          0.09983342  0.19866933  0.29552021  0.38941834  0.47942554
  0.56464247  0.64421769  0.71735609  0.78332691  0.84147098  0.89120736
  0.93203909  0.96355819  0.98544973  0.99749499  0.9995736   0.99166481
  0.97384763  0.94630009  0.90929743  0.86320937  0.8084964   0.74570521
  0.67546318  0.59847214  0.51550137  0.42737988  0.33498815  0.23924933
  0.14112001  0.04158066 -0.05837414 -0.15774569 -0.2555411  -0.35078323
 -0.44252044 -0.52983614 -0.61185789 -0.68776616 -0.7568025  -0.81827711
 -0.87157577 -0.91616594 -0.95160207 -0.97753012 -0.993691   -0.99992326
 -0.99616461 -0.98245261 -0.95892427 -0.92581468 -0.88345466 -0.83226744
 -0.77276449 -0.70554033 -0.63126664 -0.55068554 -0.46460218 -0.37387666]
>>> # matplotlib.pyplot 을 이용하여 데이터를 그래프로 그려준다
>>> plt.plot(x, y)	# plot 에 데이터 할당하여 그래프를 그려줌
>>> plt.show() 		# 그래프를 보여줌

>>> # 그래프 2개 그리기(sin, cos)
>>> y_1 = np.sin(x)
>>> y_2 = np.cos(x)
>>> plt.plot(x, y_1, label = "sin")
[<matplotlib.lines.Line2D object at 0x000001FFE775BBE0>]
>>> plt.plot(x, y_2, linestyle = "--", label = "cos")
[<matplotlib.lines.Line2D object at 0x000001FFE775BFD0>]
>>> plt.xlabel("x")
Text(0.5, 0, 'x')
>>> plt.ylabel("y")
Text(0, 0.5, 'y')
>>> plt.title("sin & cos")
Text(0.5, 1.0, 'sin & cos')
>>> plt.legend() 		# 좌측하단에 label ("sin", "cos") 과 선(선, --선)이 써진 작은 박스 legend 가 생성된다.
<matplotlib.legend.Legend object at 0x000001FFE774A5B0>
>>> plt.show()

### image(png) 파일 읽어오기
>>> import os
>>> os.getcwd()
'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python38'
>>> os.chdir("C:\\Users\\user\\Desktop\\")		# 경로 변경
>>> os.getcwd()
'C:\\Users\\user\\Desktop'
>>> import matplotlib.pyplot as plt
>>> from matplotlib.image import imread
>>> img = imread("image_test.PNG")
>>> plt.imshow(img)
<matplotlib.image.AxesImage object at 0x0000027AB8B88100>
>>> plt.show()
