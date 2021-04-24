---
layout: post
title: '[디지털영상처리]Introduction and Digital Image Fundamentals'
subtitle: 'Digital Image Processing, Introduction and Digital Image Fundamentals'
categories: CV
tags: 영상처리
published: true
---
>본 글은 Digital Image Processing을 학습하고 정리를 하여 남겨두기 위함입니다.  
>출처 : Digital Image Processing, 3rd Edition, Gonzalez and Wood, Pearson (2010)  
>고려대학교 윤창현 교수 디지털영상처리 강의 (<a>http://www.kocw.net/home/cview.do?mty=p&kemId=1094777</a>)  
> 수정중에 있음.  

## 서론

디지털 영상 처리의 기법

Image Enhacement : 이미지를 눈에 보기 좋게 하기 위하여 처리하는 과정.
<br/> - Enhancement에는 Spatial Domain과 Frequency Domain이 있음.
<br/> - Spatial에는 Point와 Neighborhood 방법이 있음.
<br/> - Frequency에는 Transform을 이용.

Image Segmentation : 이미지를 적당히 나눠서 컴퓨터가 처리할 수 있게 구별하고 Label을 붙이는 과정.

Image Understanding : 이미지가 무엇인지 컴퓨터가 판단하는 과정.
<br/> - Recognition
<br/> - (Artificial Inteligence 영상처리에선 논외)

## Chapter 2. Digital Image Fundamentals

### 마하밴드 효과(Mach Band)

<img src="/assets/img/영상처리/mach.png" title="Mach Band Effect"/> 

위와 같이 명암의 대비가 되는 경계선 부근에서는 사람의 눈으로 인지하기에는 경계에서 색이 더 진해보이거나 밝게 보이는 현상이 일어난다.

### Sampling and Quantization

![Sampling](/assets/img/영상처리/sampling.png)

선분 AB에 대해서 명암의 정도를 나타낸 것이다.

이 값을 유한한 개수의 데이터로 제한하여 나타내는 것을 `Sampling`이라고 한다.

이러한 샘플링한 아날로그 형태로 되어있는 데이터를 디지털화하는 것을 `Quantization`이라고 한다.

<img src="/assets/img/영상처리/샘플링2.png" align="left" title="sampling2"><br/>



Sampling과 Quantzation을 처리한 결과이다.

### Digital Image 표현

이미지는 기본적으로 M X N Matrix로 표현할 수 있다.<br/>
L은 명암의 레벨을 binary로 나타낸다.<br/>
M X N X L 이 이미지의 크기를 결정한다.<br/>

(Saturation : 채도, Intensity : 명암도)

### Spatial and Intensity Resolution

<img src="/assets/img/영상처리/화면해상도.png" align="left" title="Spatial Resolution"><br/>



이미지 픽셀의 세분화가 해상도를 결정함.

<img src="/assets/img/영상처리/밝기해상도.png" align="left" title="Intensity Resolution">
<br/>
<br/>

Contazation시 비트 수를 얼마나 세분화하느냐가 해상도를 결정함.

![해상도](/assets/img/영상처리/해상도.png)![해상도2](/assets/img/영상처리/해상도2.png)

얼굴 사진 같은 경우는 밝기해상도가 중요하고, 군중 사진과 같은 경우는 화면해상도가 중요하다.

번외로 Image Interpolation을 통해 분해능이 낮더라도 보완이 가능함.

### 인접 Pixel

#### Neighborhood
![인접1](/assets/img/영상처리/인접1.png)
<center>4-neighborhood relation</center>
![인접2](/assets/img/영상처리/인접2.png)
<center>8-neighborhood relation</center>
![인접3](/assets/img/영상처리/인접3.png)
<center>Diagonal-neighborhood relation</center><br/>
#### Distance Measure
![거리1](/assets/img/영상처리/거리1.png)
<center>일반적인 거리 계산의 방법</center>
![거리2](/assets/img/영상처리/거리2.png)
<center>City Block Distance : 4-neighborhood relation과 관련이 있다.</center>
![거리3](/assets/img/영상처리/거리3.png)
<center>Chessboard Distance : 8-neighborhood relation과 관련이 있다.</center>

###디지털 영상 처리의 Math Tools
이미지는 행렬로 표현할 수 있으므로 행렬 연산이 가능하다.

이미지의 선형 연산은 Additivity(덧셈성)와 Homogeneity(동차성)를 만족하는 Superposition이여야 선형 연산이 가능하다.

같은 공간의 다른 시간의 이미지를 평균을 하는 산술 연산 또한 가능하다. 평균을 하면 노이즈가 감소하는 효과를 보인다. 잡음 제거에서 평균을 많이 사용하는데 추후 알아보도록 하자.

또한 빼기 연산, 곱하기 연산과 같은 모든 산술 연산을 통해 영상 처리를 하는 것이 있다.

Negative 영상 처리 같은 것도 집합 연산, 논리 연산을 이용하여 하는 것이다.

공간 연산(Spatial Operation)을 통해 블러링과 같은 처리를 할 수도 있다.

벡터와 행렬 Matrix 연산을 통해 스케일링, 회전, 이동, 쏠리게 여러가지 처리를 할 수 있다.
