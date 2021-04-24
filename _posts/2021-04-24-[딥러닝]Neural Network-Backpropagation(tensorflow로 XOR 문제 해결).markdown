---
layout: post
title: '[딥러닝]Neural Network-Backpropagation(tensorflow로 XOR 문제 해결)'
subtitle: 'Neural Network, Perceptron, Backpropagation, XOR'
categories: ML
tags: 딥러닝
published: true
---
> 본 글은 강의를 수강하고 복습을 하며 추가로 공부하고 기록을 남기기 위한 글입니다.  
>   
>   
> 출처 : 인하대학교 김승환 교수 2021-1학기 데이터마이닝 수업

### 서론

[2021.04.23 - \[ML/딥러닝\] - \[딥러닝\]Neural Network - Perceptron(AND, OR, XOR)](https://maru-jang.tistory.com/5)

[

\[딥러닝\]Neural Network - Perceptron(AND, OR, XOR)

본 글은 강의를 수강하고 복습을 하며 추가로 공부하고 기록을 남기기 위한 글입니다. 출처 : 인하대학교 김승환 교수 2021-1학기 데이터마이닝 수업 서론 Neural Network는 인간의 뇌를 수학적 모형

maru-jang.tistory.com



](https://maru-jang.tistory.com/5)

이전 글에서 Multi Layer Perceptron을 Backpropagation을 통해 가중치를 구하고 Output을 예측하는 것을 해보았다.

이제 Backpropagation이 가중치를 구하는 방법은 어떻게 진행되는지 알아보고 tensorflow를 통해 다시 한번 XOR 문제를 해결해보자.

### Backpropagation

이전 글과 같은 XOR 문제를 해결하기 위해 아래와 같은 신경망을 가정하였다.

input은 2개이고, output은 1개이다.

Hidden Layer는 1개, Hidden Layer의 Node는 3개로 가정하였다. (자유롭게 가정 할 수 있다.)

w1은 2 X 3 Matrix이고, w2는 1 X 3 Matrix이다.

b1은(h = w1·x + b1) 1 X 3 Matrix, b2는(y = w2·h + b2) 1 X 1 행렬이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F1Z7vz%2Fbtq3l3VASBT%2F8jjaVz8MRkFIvfvdC43zA0%2Fimg.png" title="신경망">

여기서 h = g(x·w1), y = g(h·w2), t : Target Value 이고, 활성함수 g 는 Sigmoid 함수이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtHl4X%2Fbtq3mOcsdf7%2Fg9PGbQ9yWtunDk5zakijk1%2Fimg.png">

Cost 함수에서 1/2을 하는 이유는 Cost 함수를 미분하여 가중치를 구할 때 단순 계산의 편의를 위함이며, 최소값을 찾는 과정이기에 값에는 영향을 주지 않는다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNiQl3%2Fbtq3ncYtLn7%2FXVOCmA42qE1ZgV2LRKmG7k%2Fimg.png">

목표는 Cost 함수를 최소화하는 가중치를 구하는 것이고, 가중치는 아래와 같이 Gradient Descent Method를 반복적으로 사용하여 구한다. 여기서 lambda는 Learning Rate로 수렴속도를 조절하는 값이다.

(Gradient Descent Method(경사하강법)에 대해 간단히 짚고 넘어가자면 Cost함수를 미분하여 기울기를 구하고 기울기에 Learnig Rate를 곱한만큼 빼주어 초기 가중치(W)에서 기울기 방향으로 조금씩 이동하여 기울기가 0인 최소값을 찾는 방법이다.)

먼저 Output 값을 결정하는 w2부터 구하는 과정을 거친다. 이렇게 Output에서부터 역순으로 가는 과정때문에 **'Back + Propagation'**이라고 한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbxiIXx%2Fbtq3lTZUYdK%2FxqP7SakENEN0amxpWwsUb0%2Fimg.png">

먼저 Chain Rule에 의거하여 Cost함수를 w2에 대해 미분을 하면 위와 같은 과정을 통해 구하고 이를 이용해 w2를 반복적으로 갱신을 해줄 수 있다.

다음은 w1의 차례이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcELxPZ%2Fbtq3nixsVxs%2FPZyW582kSn9aolGwby2Vvk%2Fimg.png">

위 과정에선 생략하였지만 상수항과 같은 b에 대한 업데이트 식은 b = b - lambda · deltaH가 된다.

즉, Backpropagation은 앞서 구한 delta 값을 이용하여 다음의 delta 값을 계산하는 원리로 Output Layer에서 Input Layer로 역 과정을 통해 가중치를 보정 전파 할 수 있음을 보인다. 이러한 과정을 통해 여러 층의 Hidden Layer도 문제없이 구할 수 있게 된다. 예시로 Hidden Layer가 3개인 Backpropagation을 아래에서 확인 할 수 있다.

https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdKa3b%2Fbtq3nI3KaBZ%2F3bUV3CBNK6kkKQIb86FaK1%2Fimg.png

이제 이 전 포스팅에서의 XOR 문제 해결에 대해 이해가 됐을 것이다.

이번 포스팅에서는 tensorflow를 이용하여 XOR 문제를 Backpropagation 알고리즘으로 해결해보자.

```python
import tensorflow as tf
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(3, activation='sigmoid', input_dim=2))
model.add(layers.Dense(1, activation='sigmoid'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=1000, batch_size = 4, verbose = 1)
model.evaluate(x, y)

```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcARRCj%2Fbtq3mpDZZe6%2FZelhqWI6doLXxZrWAHaEDk%2Fimg.png">

learnig rate : 0.5, epochs : 1000회, batch size : 4 일 때 loss 0.0626, accuracy 1.0이 나왔다.

loss에 대해 언급만 하고 넘어가자면 loss가 0으로 가까워질 때까지 학습할수록 정확도가 높아진다.

```python
predicted = model.predict(x)
print(predicted)
```

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Foe2YL%2Fbtq3mwXiQHB%2FsTH9IRzSPU24OpijDqGYK1%2Fimg.png">

XOR은 간단한 모델로 따로 test set이 존재하지 않아 train set으로 test를 해보았다.

테스트 결과 XOR Target Value와 근사한 결과가 나옴을 확인하였다.

이후 공부해 볼 것은 XOR을 Hidden Layer와 그에 따른 Node를 많이 추가하면 왜 원하는 결과를 얻을 수 없는지, 그리하여 어떤 해결책(ReLU 함수)이 있는지를 공부해보도록 하자.