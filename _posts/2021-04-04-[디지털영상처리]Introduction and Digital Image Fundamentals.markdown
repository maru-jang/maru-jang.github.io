---
layout: post
title:  "[디지털영상처리]Introduction and Digital Image Fundamentals"
subtitle:   "Digital Image Processing, Introduction and Digital Image Fundamentals"
categories: CV
tags: 영상처리
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

<img src="/assets/img/영상처리/샘플링2.png" align="left" title="sampling2">
Sampling과 Quantzation을 처리한 결과이다.

### Digital Image 표현

이미지는 기본적으로 M X N Matrix로 표현할 수 있다.<br/>
L은 명암의 레벨을 binary로 나타낸다.<br/>
M X N X L 이 이미지의 크기를 결정한다.<br/>

Saturation, Noise...<br/>
`수정중`<br/>

### Spatial and Intensity Resolution

<img src="/assets/img/영상처리/화면해상도.png" align="left title="Spatial Resolution">
이미지 픽셀의 세분화가 해상도를 결정함.

<img src="/assets/img/영상처리/밝기해상도.png" align="left" title="Intensity Resolution">
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




>수정중
>에버리징하면 노이즈가 줄어들어 .
>왜 사진은 2배3배되고 노이즈는 루트2배루트3배돼.
>스파셜 오퍼레이션, 프로세싱
>싱글 픽셀 오퍼레이션 : 밝기 거꾸로.
>네이버후드 오퍼레이션 : 인접픽셀 사용해서 계산. 예시는 인접픽셀 에버리지해서 뿌얘져??????
>지오메트릭 스파셜 트랜스폼 and 이미지 레지스트레이션
> :스케일링. 로테이션 트랜스레이션 쉬어 쉬어 버티컬 호라이즌






