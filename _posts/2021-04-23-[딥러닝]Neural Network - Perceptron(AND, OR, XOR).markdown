---
layout: post
title: '[딥러닝]Neural Network - Perceptron(AND, OR, XOR)'
subtitle: 'Neural Network, Perceptron, Backpropagation'
categories: ML
tags: 딥러닝
published: true
---

> 본 글은 강의를 수강하고 복습을 하며 추가로 공부하고 기록을 남기기 위한 글입니다.  
>   
> 출처 : 인하대학교 김승환 교수 2021-1학기 데이터마이닝 수업

### 서론

Neural Network는 인간의 뇌를 수학적 모형으로 표현하여 인간처럼 판단을 수행하고자 하는 아이디어로부터 출발하였다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FevK7Iq%2Fbtq3lTqpz0m%2FViH901AxhXGwU287lwNQE1%2Fimg.png" title="신경망1">

여러 개의 가중치와 여러개의 Input (x) 값을 통해 output (y) 값이 나오도록 가중치(w)를 구하고자 하는 것이다.

Input (x) 값에 각 가중치 (w) 를 곱하는데 가중치가 클수록 그 Input값이 중요하다는 의미이다.

이러한 함수 f를 활성함수(Activation fuction) 라 한다.
<figure>
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwmSYi%2Fbtq3h6KUyDw%2Fz8Wf8oikK2A3ivpIHvIkqk%2Fimg.png" title="신경망2">    
    <figcaption>y = f(&sum;WiXi+b)</figcaption>
</figure>

활성함수를 Sigmoid로 사용 할 경우, 신경망 모형은 Logistic Regression 모형이 된다.

<figure>
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFR3KG%2Fbtq3hu6AIX6%2FY7sKc5yE6ieqs0yaVi27dk%2Fimg.png"> 
    <figcaption>활성함수가 Sigmoid인 Nerual Network 모형</figcaption>
</figure>

### Perceptron - OR, AND, XOR

초기 Nerual Network(신경망) 모형은 Linear한 모형으로도 or, and 문제가 해결이 가능하나 xor 문제는 해결이 불가능하여 xor 문제를 어떻게 푸는냐가 관건이였다.
https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb5HDFy%2Fbtq3htGo8hS%2F1E53t4Fmmgp7DOBBZeaVz0%2Fimg.png

1969년 Hidden Layer를 사용하는 Perceptron을 사용해 xor 문제를 풀 수 있음을 증명하였는데 문제는 다층 perceptron(Multiple Layer Perceptron, MLP)의 가중치(w)는 어떻게 구하냐는 것이다.

<figure>
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FsPFki%2Fbtq3itlJui4%2F9lHKxliB4LvOgNVer1DQB0%2Fimg.png">
    <figcaption>두 개의 Linear 모형으로 XOR 문제 해결</figcaption>
</figure>

1979년 Backpropagation 알고리즘이 나와 다층 perceptron의 가중치를 구할 수 있게되었다.

**그렇다면 먼저 or, and 문제를 단층 perceptron으로 해결해보자.**

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3Lnc8%2Fbtq3mNJWs2n%2FMAPbnfxxJUhrp01VbUEkq1%2Fimg.png">
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbjpsok%2Fbtq3h5MhldG%2FSEYXAWuZ57sox8ut98gNjK%2Fimg.png">
이러한 Neural Network 모델이 있을 때 우리는 가중치를 구해보아야 한다.

아래 식에서 t는 true 값(output의 실제값)이고 f(net)은 네트워크를 통해 계산된 y값(output 추정값)이다.

t - f(net)은 오차가 된다.

w는 아래의 식으로 t - f(net)의 오차가 양수이면 f(net)이 커져야 하므로 가중치에 일정량을 더하고 음수면 f(net)이 작아져야 하므로 가중치에 일정량을 뺀다.

여기서 에타는 상수로 학습률(Learning Rate)라고 부른다.

학습률은 w값이 목적값으로 가는 속도를 조절하는 상수역할을 하는데 에타가 크면 빨리 해로 가지만 정확한 해를 구하기 어렵고 에타가 작으면 해로 느리게 가지만 정교한 해를 구할 수 있다.

<figure>
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbCYuXs%2Fbtq3hvK9Etc%2F7sah0sN4rug6mAQvxKwImK%2Fimg.png">
    <figcaption>단층 perceptron</figcaption>
</figure>

```python
import numpy as np

def fnet(net):
  if net <= 0:  # output이 0보다 작으면 0
    return 0
  else:         # output이 0보다 크면 1
    return 1

eta = 0.1       # 학습률은 임의로 0.1로 둠

x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])  # input
t = np.array([0,1,1,1])      # AND output
w = np.array([0.5,0.5,0.5])  # 초기 가중치는 임의로 0.5로 둠

for j in range(10):  # 10회 학습 (epoch)
  for i in range(4):  # 4가지 케이스 (batch size)
    y = fnet(np.dot(w, x[i]))
    w[0] = w[0] + eta * x[i][0] * (t[i]-y)
    w[1] = w[1] + eta * x[i][1] * (t[i]-y)
    w[2] = w[2] + eta * x[i][2] * (t[i]-y)
  print(w)
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBCIHx%2Fbtq3mZjjfaJ%2Fut0ZiPVRj9zxLZ2pNRo0F1%2Fimg.png"><br/>
매우 간단한 모델이므로 6회의 학습만으로도 최적의 가중치를 찾아낸 것을 확인 할 수 있다.

이 가중치를 이용하여 True 값과 학습된 가중치로 얻은 output 값이 같은지 확인해보자.

```python
x = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
w = np.array([-0.1,0.5,0.5])
y = np.array([])

for i in range(4):
  y_i = fnet(np.dot(w, x[i]))
  y = np.append(y, y_i)
print(y)
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F9bZaY%2Fbtq3lSyyaoa%2FQkWHfTn9QOfAG9JkSIVkRK%2Fimg.png">

위는 AND를 해결한 것으로 같은 방법으로 OR 문제도 해결 할 수 있다.

이렇게 단층 perceptron으로 가중치를 구하여 and, or 문제를 해결함을 보였다.

**이제 hidden layer를 추가하여 다층 Perceptron을 Backpropagation으로 xor 문제를 해결해보자.**

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbB6X9w%2Fbtq3iLfMLr0%2Fp8AeNVBl8ZUwnBFgWyiYqk%2Fimg.png"><br/>

XOR 문제는 먼저 input 값이 2개이고 output 값이 1개이다. 히든레이어는 한개, 노드는 3개로 임의로 정하였다.

w1은 총 6개의 선, 2 X 3 Matrix를, w2는 3개의 선, 3 X 1 Matrix를 형성하고 있다.

b1은 1 X 3 Matrix이고, b2는 1 X 1 Matrix 이다.

H = x \* w1 + b1이고, output = H \* w2 + b2 로 계산한다.

```python
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([[0,0],[0,1],[1,0],[1,1]])
t=np.array([[0],[1],[1],[0]])
print("x : \n", x)
print("True : \n", t)

w1=2*np.random.rand(2,3)-1  # np.random.rand : 2*(0~1) 난수 -1 : -1 ~ 1
b1=2*np.random.rand(1,3)-1  # 가중치를 임의로 M X N Martix에 두기 위함.
w2=2*np.random.rand(3,1)-1

print("w1 : \n", w1)
print("b1 : \n", b1)
print("w2 : \n", w2)
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcrqF2z%2Fbtq3itzCkhi%2FAieiEougC5C9wnY3zqeO3k%2Fimg.png" title="xor2">
x와 실제 output 그리고 임의의 가중치 값 들을 확인하였다.

```python
h=sigmoid(np.dot(x,w1)+b1)
y=sigmoid(np.dot(h,w2))
y
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcrqF2z%2Fbtq3itzCkhi%2FAieiEougC5C9wnY3zqeO3k%2Fimg.png">
output이 아직은 엉터리로 나오는 것을 볼 수 있다.

```python
lamda = 1
for i in range(1000):  # epoch : 1000
    h = sigmoid(np.dot(x,w1) + b1)
    y = sigmoid(np.dot(h,w2))
    deltaY= np.multiply(y - t,np.multiply(y,(1-y)))
    temp = np.multiply(w2.transpose(),np.multiply(h,(1-h)))
    deltaH = deltaY * temp
    w2=w2-np.dot(h.transpose(),lamda*deltaY)
    w1=w1-np.dot(x.transpose(),lamda*deltaH)
    b1=b1-lamda*deltaH
print (y)
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F449iy%2Fbtq3mcRcopp%2FrVz10Gy0b3kcD8S53bGeO1%2Fimg.png">
학습된 가중치로 output을 구하였더니 XOR 실제 outpu 값인 0, 1, 1, 0과 근사한 값을 얻어낼 수 있었다.

위는 BackPropagation으로 해결한 것으로 아직 Backpropagation에 대해 증명을 하지 않았으므로 설명은 생략하도록 하겠다.

번외로 이러한 Neural Network의 단점으로는 over fitting과 layer의 수가 커질수록 학습이 안된다는 문제가 있었다.

2006년 이후 이 단점에 대한 솔루션들이 많이 나오고 다시 딥러닝이라는 이름으로 세상에 알려지게 되었다.

이후 포스팅은 Backpropagation을 증명하며 위 XOR 문제를 다시 상기시켜보도록 하자.
