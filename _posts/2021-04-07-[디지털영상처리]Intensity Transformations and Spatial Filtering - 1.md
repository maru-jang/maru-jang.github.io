---
published: false
---
> 본 글은 디지털영상처리를 공부한 기록입니다.  
> 출처 : 대학공개강의 전남대학교 홍성훈 교수 디지털영상처리 강의 ([www.kocw.net/home/cview.do?mty=p&kemId=320576)](http://www.kocw.net/home/cview.do?mty=p&kemId=320576)  
> Digital Image Processing, 3rd Edition, Gonzalez and Wood, Pearson (2010)  


※용어 설명

Restoration 원본에 가깝게

Enhancement 사람 눈에만 잘 보이게

## Background

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlOM9t%2Fbtq16RUkl1x%2Fk1cw4mc8ILG7QLishyL6t0%2Fimg.png"/>

f(x,y) :  input image

g(x,y) : output image

T\[  \] : operator (변환)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fozw8Q%2Fbtq1ZEW0mww%2FfZKiRtSuI9lALx6NxxG6NK%2Fimg.png"/>

왼쪽과 같은 경우는 Contrast-stretching으로 input image의 k 밝기 이하는 어둡게 k 밝기 이상은 밝게 만들어서 원래 영상보다 높은 Contrast(대비)를 갖는 영상을 만들게 되는것이다.

오른쪽의 경우에는 k를 기준으로 이진으로 매핑 하는 것을 Thresholding function이라 한다.

매우 단순하고 강력한 처리를 행한다.

이로써 기본적인 밝기 변환에 대해서 알아보았다.

## Basic Intensity Transformation Functions

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fz5XPe%2Fbtq16r2Fvnn%2F3CZSmfXEDXTFT4loKpg3H1%2Fimg.png"/>

위 사진은 기본적인 밝기 변환 함수들을 나타내는데, log 변환은 어두운 레벨에 분포된 픽셀들을 넓은 범위의 밝기로 표현하여 어두운 쪽을 잘보이게 하는 변환이며, 역 log 변환은 그와 반대로 밝은 쪽을 잘보이게 하는 변환이다.

#### Negative Trasformations

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHkXsD%2Fbtq14Y0WyDQ%2FH5Jmkd9NjXR9EgLoPcwhKk%2Fimg.png"/>

r : input
s : output

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJEttR%2Fbtq16zsLC9B%2FMI0KXr0Cp9AjrbtqdaMwX1%2Fimg.png"/>

위와 같은 방식으로 밝기 레벨을 반전시킴으로써 영상의 어두운 영역에 놓여있는 흰색이나 그레이 디테일을 개선 시키는 데 적합하다.

위의 가슴 X-ray 사진의 예시와 같이 Negative Transform을 통해 조직을 분석하는 것이 더욱 쉬워진다.

#### Log Transformations

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHEoD8%2Fbtq1XTtl1TP%2FGv4meIf27HKJyA3pUWnSr0%2Fimg.png"/>

앞선 Log 변환의 설명을 잘 보여주는 예시이다.

고주파쪽의 에너지가 적어서 어두운 부분이 잘 안보이는 현상이 나타난다.

따라서 어두운쪽을 Stretching 시켜야 한다.

#### Power-Law(Gamma) Transformations

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcIJ0yF%2Fbtq1ZFBCkS3%2F3XpFmArYUbHq2oiM7CQ3t1%2Fimg.png"/>

input image에 gamma 제곱을 하는 변환으로, Gamma가 1보다 커지면 밝은 쪽이 Stretching되어 잘보이게 되어지게 된다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbHN1Te%2Fbtq12l98b0Q%2FjDl2lSYDOS93ihkzsGRNP0%2Fimg.png"/>

이에 대한 예시로 어떠한 화면이 TV모니터, 스크린 상으로 보게 된다면 퇴색해 보이거나, 어두워 보일 수 있다.

이 때문에 적절한 Gamma 보정이 필요하다.

Gamma 보정을 거쳐 Original Image와 비슷하게 보일 수 있다. 이렇게 Gamma 보정은 Contrast Enhancment에 유용하다.

추가로 Gamma보정은 적, 녹, 청의 비도 바뀌기 때문에 컬러를 정확하게 재현하기 위해서는 Gamma 보정에 주의가 필요하다.

#### Piecewise-Linear Transformations

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdgGFz8%2Fbtq16Aee0Qc%2F4h5rov5nOpbx3QQE1iisFk%2Fimg.png"/>

그림(a)에서 r1~r2의 input 값을 더욱 잘 보이게 하는 Stretching을 하는 결과 (b)가 (c)와 같이 보이게 된다.

(d)는 thresholding 처리한 것으로 극단적인 예이다.

이렇게 구간을 정하여 선형으로 변환하는 경우도 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FDduDk%2Fbtq11M08xUw%2FGSmnYATnSgXn6H3wKdJWeK%2Fimg.png"/>

왼쪽 그림과 같은 구간 변환은 AB구간만 남기고 안보이게 하는 변환.

오른쪽은 원본 이미지와 같게 하되, AB구간을더욱 잘보이게 하는 변환이다.

아래는 순서대로 원본, 왼쪽과 같은 변환, 오른쪽과 같은 변환을 나타낸 그림이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbN0VDH%2Fbtq16yOgohY%2Fe352vu45ogkpkWqZq2lppK%2Fimg.png"/>

#### Bit-Plane Slicing

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtNv4Z%2Fbtq12651uol%2FNNKDpv0UjWLko5X68ViII0%2Fimg.png"/>

8bit 영상은 1비트 평면 8개로 구성되어 있다.

필요한 Bit-Plane의 Scale만 뽑아서 처리할 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvXPrz%2Fbtq11yPuNtX%2F8v6muymEKRiWj8clQook01%2Fimg.png"/>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3Gcju%2Fbtq11AfvsST%2FyAcjJEl5chMapWxy0QXEOk%2Fimg.png"/>

그림에서 볼 수 있듯이 중요도는 MSB에 가까울수록 중요함을 확인 할 수 있고, 8, 7 Bit Plane만 있어도 원본과 비슷함을 알 수 있다.

하지만 LSB에 가까운 Plane도 영상이 단조로워 보이지 않기 위해서 필요한 Bit-Plane들이다.

## Histogram Processing

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbxLhP5%2Fbtq16QnG9pH%2FpUPKZlwjWI4SKh0vODPDDK%2Fimg.png"/>

영상의 밝기 값들의 빈도 수를 카운트해서 이산 함수로 나타낸 것이 히스토그램이다.

이를 전체 화소수로 나누어서 정규화를 시킬 수 있다. (PDF, 확률밀도함수)

이러한 과정을 직접 코딩을 통해 해보도록 하는 시간을 가져보자.

이러한 히스토그램을 보고 Equalization(평활화) 처리를 함으로써 이미지를 개선시킬 수 있다.

s(output)은 uniform 해야 한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FXfIme%2Fbtq1ZGAAnkQ%2FhZr4ri1vQjixVLXuLKKzPK%2Fimg.png"/>

복잡한 계산 과정이 있지만 생략하고 간단하게 설명하자면 Histogram Equalization은 CDF를 가져다가 적절히 맵핑을 한다면 uniform한 Histogram과 함께 개선된 image가 확인됨을 알 수 있다.

1번의 예시로만 설명하자면 모든 화소가 어두운 쪽의 화소에 집중 되어 있고 이것 만을 맵핑시키면 됨을 알 수 있다.

간단히 설명했지만 복잡한 계산 과정이 있으므로 추후 궁금하다면 다시 한번 찾아보자.

다음 시간에 히스토그램과 Spatial Filtering(공간 필터링)을 이어서 공부해보도록 하자.