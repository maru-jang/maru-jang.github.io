---
layout: post
title: '[딥러닝]Tensorflow-MNIST Data'
subtitle: 'tensorflow,mnist'
categories: ML
tags: 딥러닝
published: true
---

>본 글은 강의를 수강하고 복습을 하며 기록하기 위한 글입니다.<br/>
>출처 : 인하대학교 김승환 교수 2021-1학기 데이터마이닝 수업<br/>
>수정중

### 서론

딥러닝의 Hello, World라고 할 수 있는 mnist Dataset을 tensorflow를 통해 학습을 해보면서 one-hot encoding, GradientDescent알고리즘, sigmoid 함수를 좀 더 피부에 와닿게 이해해보도록 할 수 있을 것이다.<br/>
<img src="/assets/img/딥러닝/mnist1.PNG" title="mnist"/>
이처럼 사람마다 필체가 다른데 이러한 수기로 쓴 숫자를 판별할 수 있는 모델을 만드려고 한다.<br/>
MNIST Dataset을 이용하여 tensorflow로 학습시키고 숫자를 판별할 수 있는 로지스틱 모형을 개발해보자.<br/>

### MNIST Data
tensorflow를 통해 mnist 데이터셋을 받아오도록 하자.
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
```
데이터셋의 shape을 확인해보면
```python
print(train_x.shape, test_x.shape)
print(train_y.shape, test_y.shape)
print(np.max(train_x[0]), np.min(train_x[0]))
```
training x 데이터셋은 60000X28X28의 3차원 Matrix로<br/>
test x 데이터셋은 10000X28X28의 3차원 Matrix로 확인됨을 알 수 있다.<br/>
자료는 서론에서의 그림과 같이 28*28의 픽셀에 0~255사이의 숫자로 표현된 형식임을 확인했다.<br/>
<br/>
이제 하나의 이미지를 1~784의 독립변수로 만들고 표준화를 해준다.
```python
train_x = train_x.reshape(-1,28*28) 
test_x = test_x.reshape(-1,28*28)

train_x = train_x / 255 # 표준화
test_x = test_x / 255
```
그리고 이에 대한 레이블을 one-hot encoding 기법을 이용하여 0~9로 입력받아 학습을 수행할 것이다.
```python
train_y_onehot = to_categorical(train_y)
test_y_onehot = to_categorical(test_y)
```
이제 학습을 시작할 준비를 해보자.
```python
from tensorflow.keras import layers
model = tf.keras.Sequential()
model.add(layers.Dense(10, activation='softmax', input_dim=784))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
```
```python
model.fit(train_x, train_y_onehot, validation_data=(test_x, test_y_onehot), batch_size = 100, epochs=5)

model.evaluate(train_x, train_y_onehot)

model.evaluate(test_x, test_y_onehot)

predicted = model.predict(test_x)

predicted[0], test_y_onehot[0]
```
`수정중`
