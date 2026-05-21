# Leaffliction — Defense Script

평가표 순서대로 시연하면서 입으로 읽을 수 있게 만든 대본. 각 섹션은 (1) 평가자가 보는 것, (2) 명령어, (3) 입으로 할 말, (4) 관련 코드, (5) 함정으로 구성됨.

---

## 0. 프로젝트 소개

🎤 **시작할 때 평가자에게**

> "Leaffliction은 사과와 포도 잎 사진을 보고 어떤 질병인지, 아니면 건강한 잎인지 자동으로 분류하는 컴퓨터 비전 프로젝트예요. 사과 4종(healthy, Black_rot, rust, scab) + 포도 4종(healthy, Black_rot, Esca, spot) 합쳐서 총 8개 클래스고요, subject가 제공한 데이터셋에서 7,200장 정도를 받았어요. 사진은 통제된 회색 배경에 잎 1장씩 찍혀있는 학습용 데이터예요."
>
> "전체적으로는 4 Part로 구성되어 있어요. Part 1이 데이터셋을 분석하는 EDA고, Part 2가 데이터 증강, Part 3이 잎의 특징을 시각화하는 plantCV transformation, Part 4가 실제로 CNN을 학습시키고 예측하는 단계예요. 그러니까 머신러닝 프로젝트의 전체 흐름 — 데이터 보기, 전처리, 모델 학습, 추론 — 을 다 다루는 입문 과제라고 생각하시면 돼요."

🎯 핵심 설계 결정 4가지 (질문 들어오면 답할 수 있게):
- 모델은 직접 만든 ScratchCNN을 메인으로, EfficientNet-B0 transfer는 비교용
- 학습은 원본 `images/`만 쓰고, augmentation은 메모리에서만 (데이터 누수 회피)
- 클래스 불균형은 `WeightedRandomSampler`로 batch 단위에서 보정
- 모든 random은 `seed=42`로 고정해서 재현 가능

---

## 1. Error Management — Signature & Norm 검사

### 🎯 평가자가 확인하는 것
`signature.txt`의 해시가 `.zip` 파일의 SHA1과 일치하는지 `diff`로 비교. Python이면 `flake8`로 norm 검사. **불일치면 평가 즉시 종료**.

### 💻 명령
```bash
make verify                  # signature.txt vs zip 해시 자동 diff
flake8 src tests *.py        # norm
```

### 🎤 대사

> "먼저 signature 검증부터 할게요. `signature.txt`라는 파일이 있는데, 학습이 끝났을 때 두 zip 파일 — `trained_models.zip`이랑 `augmented_directory.zip` — 의 SHA1 해시를 자동으로 기록해놓은 거예요. 평가일에 zip을 USB로 옮기는 과정에서 누가 파일을 바꿔치기했는지 확인하는 단계고요, PDF Chapter V가 '0점 사유'로 명시한 부분이에요."
>
> "SHA1은 cryptographic hash라서 zip 파일 1 byte만 달라져도 완전히 다른 해시가 나와요. 그래서 신뢰성 있게 무결성 검사가 됩니다. `make verify`가 지금 그걸 한 거예요. 차이 없이 깔끔하게 통과했어요."
>
> "이어서 `flake8`로 코드 스타일 검사. 42는 평가표가 norm을 강제하기 때문에 PEP8 위반이 한 줄이라도 있으면 0점이에요. pre-commit hook에도 ruff랑 flake8 둘 다 걸어놨고, 통과 확인됐어요."

### 📂 관련 코드
- `src/leaffliction/signature.py` — SHA1 계산 + signature.txt 생성/검증. `hashlib`만 씀.
- `scripts/verify.sh` — `make verify`가 내부적으로 호출하는 셸 스크립트.
- `Makefile` — `lint`, `verify`, `test`, `format` 단축 명령.

---

## 2. Part 1 — Distribution (데이터셋 분포 분석)

### 🎯 평가자가 확인하는 것
`./Distribution.py ./images`를 실행하면 pie chart가 떠야 함.

### 💻 명령
```bash
./Distribution.py ./images
```

### 🎤 대사

> "Part 1은 EDA, 그러니까 데이터를 모델에 넣기 전에 일단 들여다보는 단계예요. ML 프로젝트의 첫걸음이고, 데이터 분포를 알아야 그 다음 모델 설계나 augmentation 전략을 결정할 수 있거든요."
>
> "`Distribution.py`는 `images/` 안의 하위 폴더 하나하나를 클래스로 보고 사진 개수를 세서 pie chart랑 bar chart 두 개로 보여줘요. pie는 비율을 한눈에 보기 좋고 bar는 정확한 개수를 보기 좋아요. 막대 위에 숫자도 같이 표시해놨고요."

차트 뜨면:

> "보시다시피 8개 클래스 합쳐서 7,228장이에요. 그런데 `Apple_healthy`가 1,640장으로 가장 많고 `Apple_rust`는 275장으로 가장 적어요. **약 6배 차이**가 나죠. 이게 클래스 불균형이고, Part 2 data augmentation이 왜 필요한지 보여주는 출발점이에요. 불균형 그대로 학습하면 모델이 다수 클래스에 편향되거든요."

### 📂 관련 코드
- `src/leaffliction/dataset.py::discover_classes` — `pathlib.rglob("*.JPG")`로 모든 사진 찾아서 부모 폴더 이름(=클래스)으로 그룹핑. 결과는 dict. 같은 함수가 Part 2, Part 4에서도 재사용돼서 클래스 라벨이 어디서나 일관됨.
- `src/leaffliction/viz.py::plot_distribution` — matplotlib + seaborn으로 pie + bar를 한 figure에 배치. bar 위에 `ax.annotate`로 숫자 표시.

---

## 3. Part 2 — Augmentation (한 장에 6 변형)

### 🎯 평가자가 확인하는 것
한 사진을 넣으면 sibling 6개가 생성되어야 함. 파일명이 `image (1)_Flip.JPG`, `_Rotate.JPG`, `_Skew.JPG`, `_Shear.JPG`, `_Crop.JPG`, `_Distortion.JPG`.

### 💻 명령
```bash
./Augmentation.py "images/Apple_healthy/image (1).JPG"
ls "images/Apple_healthy/image (1)"*
```

### 🎤 대사

> "Part 2는 data augmentation이에요. 데이터 증강이라고도 하고요. 같은 잎 사진을 좌우로 뒤집어도 같은 질병이잖아요. 회전시켜도 마찬가지고요. 그러니까 변형된 사진도 똑같은 라벨로 학습 데이터에 추가하면 데이터가 부족할 때 부풀려서 쓸 수 있고, 모델이 더 robust하게 학습돼요."
>
> "PDF가 명시한 6가지 변형 — Flip, Rotate, Skew, Shear, Crop, Distortion — 을 한 사진에 각각 적용해서 6개의 sibling 파일을 만들어요. Albumentations 라이브러리로 구현했어요."

`ls` 결과 보여주며:

> "보시면 원본 옆에 `image (1)_Flip.JPG`처럼 suffix만 붙은 6개 파일이 생겼어요. PDF 파일명 규칙이랑 정확히 일치해요."

6 변형 의미가 궁금하다고 하면:

> "Flip은 좌우 반전. Rotate는 ±30도 회전이고, 빈 공간은 검은색으로 채워요. Skew는 x랑 y 양방향으로 평행사변형처럼 찌부러뜨리는 거고, Shear는 x축으로만 더 강하게(±25도) 기울이는 거예요. Crop은 사진의 70~100% 영역을 잘라서 다시 256×256으로 확대해요. 그래서 Crop 결과물은 잎이 화면을 좀 더 채우게 보여요. 마지막 Distortion은 카메라 렌즈 왜곡을 흉내내는 건데, 잎에 직선이 거의 없어서 시각적으로는 잘 안 보이지만 픽셀 좌표는 미세하게 흔들려요."

> "이 6개가 어떻게 다른가는 사실 사람 눈에 비슷해 보일 수도 있는데요, 핵심은 사람한테 안 보여도 모델 입장에선 픽셀이 다 다르게 들어온다는 거예요. 그래서 학습할 때 같은 사진을 외우지 않고 더 일반화된 패턴을 배우게 돼요."

시연 끝나면:
```bash
rm "images/Apple_healthy/image (1)_"*.JPG
```

> "다음 단계 영향 없게 6 sibling은 정리할게요."

### 📂 관련 코드
- `src/leaffliction/augment.py`
  - `AUGMENTATION_OPS` — 6 op을 dict로 정의. key가 suffix 이름(`Flip`, `Rotate` 등), value가 Albumentations transform 객체.
  - `apply_op(name, image)` — 단일 op을 numpy 배열에 실행해 변형된 배열 반환.
  - `balance_directory(...)` — 배치 모드 핵심. 원본 복사 + 부족분만큼 augment + zip까지 자동.

### ⚠️ 함정
파일명이 정확히 `_Flip.JPG`, `_Rotate.JPG` 등 PDF 예시와 일치해야 함. 우리는 일치.

---

## 4. Part 1 추가 검증 — augmented_directory 균형 ⚠️ 0점 함정

### 🎯 평가자가 확인하는 것
같은 `Distribution.py`를 `augmented_directory`에 실행. **pie 조각이 8개 다 균등해야 함**. 아니면 Part 1과 Part 2 동시에 0점.

### 💻 명령
```bash
# zip이 아직 안 풀려있다면
unzip -q augmented_directory.zip

./Distribution.py ./augmented_directory
```

### 🎤 대사

> "이번엔 같은 `Distribution.py`를 `augmented_directory`에 실행할게요. 평가표가 명시적으로 경고한 부분인데, 여기서 pie chart가 원본이랑 똑같이 불균형하면 Part 1이랑 Part 2 둘 다 0점이라고 적혀 있어요."

차트 뜨면:

> "보시면 8 조각이 정확히 균등해요. 8 × 1,640 = 13,120장. 가장 컸던 `Apple_healthy` 1,640장에 맞춰서 나머지 7 클래스를 augmentation으로 1,640까지 채운 결과예요. 각 클래스가 정확히 12.5%."

> "참고로 이 `augmented_directory`는 PDF 요구사항 때문에 만들어두는 거지, 실제 학습엔 안 써요. 학습에 쓰면 데이터 누수 문제가 생기거든요. 이건 Part 4 모델 설명할 때 자세히 말씀드릴게요."

### 📂 관련 코드
- `src/leaffliction/augment.py::balance_directory` — 알고리즘:
  1. `discover_classes`로 클래스별 사진 모음.
  2. `target = max(...)` — 가장 큰 클래스 크기 계산.
  3. 각 클래스: 원본 복사 → 부족분만큼 6 op 중 random 적용 → 채움.
  4. 끝에 `zip_directory()` 자동 호출 → `augmented_directory.zip` 생성 (signature 비교용).

---

## 5. Part 3 — Transformation (plantCV로 잎 특징 시각화)

### 🎯 평가자가 확인하는 것
한 사진을 넣으면 6개 변환 + 9채널 color histogram이 떠야 함. 각 변환에 대해 설명 요구할 수 있음.

### 💻 명령
```bash
./Transformation.py "images/Apple_healthy/image (1).JPG"

# 옵션 확인 (PDF가 -h 명시)
./Transformation.py -h

# 배치 모드
./Transformation.py -src images/Apple_healthy -dst /tmp/out -mask
```

### 🎤 대사

#### Part 3을 왜 하는 건지부터

> "Part 3은 plantCV로 잎의 특징을 시각적으로 뽑아내는 단계예요. 사실 이게 **딥러닝 이전 컴퓨터 비전이 했던 방식**이에요. CNN이 나오기 전엔 사람이 '잎 면적, 색 분포, 모양 keypoint' 같은 feature를 수동으로 추출해서 SVM이나 Random Forest 같은 전통 분류기에 넣어서 분류했거든요. CNN의 혁신은 그 feature 추출까지 자동화한 거고요."
>
> "그러니까 Part 3은 옛날 방식을 보여주는 동시에 두 가지 부수 역할이 있어요. 첫째, plantCV라는 식물 영상 처리 전용 도메인 라이브러리를 한 번 경험해보는 것. 농학·생물학 연구자들이 실제로 쓰는 도구예요. 둘째, **데이터 sanity check** — mask가 잘 잡히는지, 클래스별 색 분포가 정말 다른지 시각적으로 확인하는 거예요."

#### 6 패널 (화면 뜨면 하나씩 짚으며)

> "Original은 원본 RGB 그대로고요."
>
> "Gaussian blur는 3×3 가우시안 필터로 작은 노이즈를 평활화해요. 다음 단계 mask를 깔끔하게 잡으려고 전처리하는 거예요."
>
> "Mask가 이 중에서 제일 핵심이에요. 잎이랑 배경을 분리하는 binary mask를 만드는데, 이게 잘 안 잡히면 다음 단계가 다 망가져요."
>
> "ROI는 mask 위에 경계 박스를 올린 거고, Analyze object는 잎 윤곽을 분석해서 면적, 둘레, 중심, 종횡비 같은 metric을 표시해요. 마지막 Pseudolandmarks는 잎 가장자리 따라 자동으로 keypoint를 잡는 건데, 모양 비교에 쓰는 표준 도구예요."

#### Mask 깊이 물으면

> "mask가 좀 까다로웠어요. 처음엔 HSV 색공간의 saturation 채널로 threshold를 잡았는데, healthy 사과 잎이 회색-녹색이라 saturation이 너무 낮아서 잎의 절반만 잡혔어요. 그래서 **LAB 색공간의 chroma magnitude**로 바꿨어요. LAB에서 a축은 녹-적, b축은 청-황 축이거든요. 회색 배경은 a랑 b가 둘 다 128 근처라서 chroma가 0에 가까워요. 반대로 녹색 잎이나 갈색 병변은 chroma가 큽니다. 그래서 chroma magnitude를 Otsu threshold로 자동 cutoff 잡고, morphological opening으로 잔티 제거하고, 가장 큰 connected component만 남겨서 잎 윤곽을 얻어요. 마지막에 scipy `binary_fill_holes`로 잎 안쪽 작은 구멍을 메워서 완성합니다."

#### 9채널 color histogram 설명 (아래쪽 패널 가리키며)

> "아래에 있는 게 9채널 color histogram이에요. X축이 픽셀 값 0~255이고, Y축이 그 값을 가진 픽셀이 사진에 몇 개 있는지 카운트예요. 각 곡선이 한 채널의 분포고요."
>
> "왜 9 채널이냐면, **3개 색공간을 같이 보거든요**. RGB 3 + HSV 3 + LAB 3."
>
> "**RGB**는 컴퓨터 native 색공간이에요. R, G, B 각각 빨강·녹·파랑 강도고요. 한계는 색이랑 밝기가 섞여 있다는 거예요. 조명이 바뀌면 R, G, B가 다 같이 변해서 색 자체 변화를 잡기 어려워요."
>
> "**HSV**는 색을 직관적으로 분리해요. H(Hue)는 무슨 색인지 — 0이 빨강, 120이 녹색, 240이 파랑. S(Saturation)는 얼마나 진한 색인지 — 0이 회색, 255가 완전 채도. 회색 배경은 S가 낮고 녹색 잎은 S가 높아요. V는 밝기고요."
>
> "**LAB**는 인간 시각에 perceptually uniform한 색공간이에요. L은 명도, a*는 녹-적 축(음수면 녹색), b*는 청-황 축(양수면 노랑). **녹색 잎은 a*가 음수로 쏠리고 갈색 병변은 b*가 양수로 길게 빠져요**. 저희 mask도 LAB를 쓴 이유가 이거예요."
>
> "왜 굳이 세 가지냐 — 한 색공간만으론 부족해서요. RGB는 native지만 조명에 민감하고, HSV는 색을 분리하지만 perceptually 균등하진 않고, LAB는 색 거리가 인간 인식이랑 비슷하지만 native 표현이 아니에요. 셋이 같은 사진을 다른 관점에서 묘사하고, 합치면 더 풍부한 색 정보가 돼요."
>
> "**클래스마다 분포가 시각적으로 달라요**. healthy 잎은 G 채널이 높은 peak, a* 채널은 음수 쪽으로 쏠려요. Black_rot은 갈색 병변 때문에 b* 채널이 양수로 더 길게 빠지고 L 채널에 어두운 tail이 생기고요. rust는 오렌지 점 때문에 H 채널에 30~60 부근 두 번째 peak가 생겨요."
>
> "옛날엔 이 histogram을 그대로 feature vector로 만들어서 — 예를 들어 9 채널 × 32 bin = 288차원 벡터 — SVM이나 Random Forest에 넣어서 분류했어요. 정확도는 70~85% 정도였고요. CNN은 이런 색 통계까지 내부적으로 conv block이 자동 학습해서 95~99%로 끌어올린 거예요."

### 📂 관련 코드
- `src/leaffliction/transform.py`
  - `_binary_mask(rgb)` — LAB chroma → Otsu → opening → largest CC → fill_holes 파이프라인. mask 핵심.
  - `gaussian_blur`, `mask`, `roi`, `analyze_object`, `pseudolandmarks` — 6 변환 각 함수.
  - `color_histogram(rgb)` — 9 채널 히스토그램 계산.
- `src/leaffliction/viz.py::plot_transformations` — `subplot2grid((2,6))`로 위에 6 변환, 아래에 6칸 span하는 hist 배치.

---

## 6. Part 4 (1/4) — Classification 정확도 (≥90%)

### 🎯 평가자가 확인하는 것
100장 이상 test에서 정확도 ≥ 90%.

### 💻 명령
```bash
./scripts/eval_val.py ./images
./scripts/eval_val.py ./images --model transfer
```

### 🎤 대사

> "Part 4 첫 번째는 100장 이상에서 90% 이상 정확도 요구예요. 저희는 `eval_val.py`라는 스크립트를 만들었는데요, 이게 어떻게 동작하냐면..."
>
> "`train.py`가 학습할 때 `random_state=42`로 stratified 80/20 split을 했어요. 그러니까 전체 7,228장 중에 1,445장이 val로 떼어져 있고, 모델은 학습 중에 **이 1,445장을 한 번도 본 적이 없어요**. `eval_val.py`는 같은 `seed=42`로 split을 재현해서 정확히 같은 1,445장을 다시 뽑아내고, 모델을 forward만 돌려서 정확도를 계산해요. PDF 요구 100장보다 14배 많은 표본으로 측정하는 거예요."

결과 뜨면:

> "1,442/1,445, **99.79%** 입니다. 90% 요구를 거의 10pp 상회하고요, per-class breakdown도 같이 나오는데 8 클래스 모두 99% 이상이에요."

transfer도 보여주며:

> "비교용 transfer 모델은 99.86%로 약간 더 높아요. ScratchCNN이랑 0.07pp 차이라서, 한쪽 모델의 트릭이라기보다는 데이터셋 자체가 명확해서 두 모델이 비슷하게 잘 풀어낸 거예요."

### 📌 미리 알아둘 — val에서 틀렸던 사진들

평가자가 "어떤 게 틀렸어요?"라고 물으면 즉답할 수 있게:

**ScratchCNN (3장 틀림)**:
- `Grape_spot/image (128).JPG` → Apple_scab (57.1%)
- `Apple_healthy/image (1040).JPG` → Apple_Black_rot (39.7%)
- `Apple_healthy/image (1326).JPG` → Grape_healthy (44.1%)

**TransferModel (2장 틀림)**:
- `Apple_healthy/image (1040).JPG` → Apple_scab (99.1%)
- `Grape_Black_rot/image (56).JPG` → Grape_Esca (93.7%)

> "재밌는 게 `Apple_healthy/image (1040).JPG`는 **두 모델 다 틀려요**. 데이터셋 자체의 어려운 사진이거나 라벨링 노이즈가 있는 케이스로 보여요. 그리고 ScratchCNN은 틀릴 때 39~57% confidence로 비교적 정직하게 망설이는데, TransferModel은 틀릴 때도 93~99%로 자신만만하게 틀려요. 이건 신경망 calibration의 전형적인 차이예요."

### 📂 관련 코드
- `scripts/eval_val.py` — `LeafDataset` 로드 → `train_test_split(stratify=labels, random_state=42, test_size=0.2)`로 val 인덱스 재현 → `trained_models.zip`에서 weight 로드 → forward → per-class accuracy.

---

## 7. Part 4 (2/4) — 모델 설명 ⭐ (5점 단일 항목, 가장 중요한 섹션)

설명 순서: (1) 큰 그림 → (2) 데이터 흐름 → (3) 모델 구조 → (4) 학습 루프 → (5) 누수 회피.

### 7-1. 왜 CNN을 썼나

🎤

> "사진 분류 문제니까 일단 CNN을 썼어요. Convolutional Neural Network라고 하는데요, 작은 필터로 이미지를 훑어서 패턴을 찾고, 그 결과를 여러 layer를 거치면서 점점 더 추상적인 의미로 만들어가는 신경망이에요."
>
> "직관적으로 생각하면 사람이 그림 볼 때랑 비슷해요. 처음엔 선이나 점 같은 저수준 특징을 보고, 다음엔 모서리나 텍스처, 그 다음엔 잎맥이나 반점 같은 중간 단계, 마지막으로 '이건 사과 잎이고 rust 같은 질병이다'라는 고수준 판단을 하잖아요. 저희 모델의 4개 conv block이 이 4단계 추상화를 자동으로 학습해요."
>
> "**핵심은 사람이 '무엇이 중요한지'를 정해주지 않는다**는 거예요. 어떤 특징이 분류에 도움이 되는지 모델이 데이터에서 직접 발견합니다. 이게 deep learning의 본질이에요."

### 7-2. 데이터가 어떻게 흘러가나 (학습 시작 전 단계)

#### Stratified 80/20 split + seed=42

🎤

> "데이터는 학습 시작 전에 세 단계의 전처리를 거쳐요."
>
> "첫 번째는 train/val split이에요. `sklearn.train_test_split`을 `stratify=labels`, `random_state=42`로 호출해요. stratified가 무슨 뜻이냐면, 일반 random split은 7,228장을 통째로 섞어서 20%를 떼면 운나쁘게 Apple_rust가 val에 0장 들어갈 수도 있거든요. 그럼 그 클래스 평가가 아예 불가능해져요. **stratified는 각 클래스마다 독립적으로 20%씩 뽑아요**. 그래서 모든 클래스가 train에 80%, val에 20%씩 정확히 들어가요."
>
> "결과적으로 train 5,783장 / val 1,445장으로 나뉘는데, val에 클래스별로 정확히 — Apple_Black_rot 124장, Apple_healthy 328장, Apple_rust 55장, 이런 식으로 — 비율 그대로 들어가요."
>
> "`seed=42`는 재현성을 위한 거예요. 컴퓨터의 random은 사실 결정론적인 함수예요. 같은 seed면 같은 시퀀스가 나오죠. 그래서 학습할 때 `seed=42`로 split하고, 평가할 때도 `seed=42`로 다시 split하면 **정확히 같은 1,445장이 val로 뽑혀요**. 이게 `eval_val.py`가 학습에 안 본 데이터를 정확히 재현할 수 있는 이유예요. 42 자체는 의미 없는 숫자고요, 그냥 ML 커뮤니티 관용 기본값이라 썼어요."

#### WeightedRandomSampler

🎤

> "두 번째 단계는 클래스 균형이에요. train set이 80/20으로 split됐어도 클래스 불균형은 그대로 남아요. Apple_rust는 train에 220장, Apple_healthy는 1,312장 있거든요. 그대로 학습하면 batch에 Apple_healthy가 압도적으로 많이 들어가서 모델이 다수 클래스로 편향돼요."
>
> "그래서 `WeightedRandomSampler`를 써요. 어떻게 동작하냐면 — 각 sample마다 weight를 매기는데, **weight = 1 / (그 sample이 속한 클래스의 크기)** 예요. Apple_rust의 한 sample은 weight가 1/220, Apple_healthy의 한 sample은 1/1312. 작은 클래스 sample은 weight가 6배 커요."
>
> "수학적으로 보면 한 클래스 안의 모든 sample weight를 합치면 `클래스 크기 × (1/클래스 크기) = 1`이거든요. 모든 클래스가 합쳐서 1로 같아요. 그래서 클래스 전체 단위로 보면 모든 클래스가 1/8 확률로 균등하게 뽑혀요. **batch 안에서 8 클래스가 거의 균등**해지는 거죠."
>
> "결과적으로 한 epoch 동안 Apple_rust 1장이 평균 3.3번 등장하고, Apple_healthy 1장은 0.55번 등장해요. 작은 클래스는 자주 보이고 큰 클래스는 덜 보이는 거예요. 그렇다고 디스크에 사진을 복제하지 않으니까 데이터 누수 위험도 없고요."

#### Online augmentation — Part 2랑 어떻게 다른가

🎤

> "세 번째 단계는 augmentation인데요, **이게 Part 2랑 다른 부분**이에요. 헷갈리기 쉬워서 짚고 가야 해요."
>
> "Part 2의 augmentation은 디스크에 augmented 파일을 만들어서 `augmented_directory`를 생성하는 거예요. 6 op(Flip, Rotate, Skew, Shear, Crop, Distortion)을 다 쓰고, 한 번 만들면 끝이에요. **이걸 학습에 직접 쓰면 안 돼요** — 누수 때문에. 그래서 Part 2 augmented_directory는 PDF의 Part 1 추가 검증 단계랑 signature 만들 때만 쓰고, 학습엔 안 써요."
>
> "Part 4의 augmentation은 완전히 다른 거예요. **메모리에서만, 학습 batch 만들 때마다 그때그때** 변형을 적용해요. op도 2개만 써요 — `RandomHorizontalFlip(p=0.5)`랑 `RandomRotation(15°)`. 6 op 전부가 아니라 가장 효과적인 두 개만요. Skew/Crop/Distortion까지 다 넣으면 학습 noise가 너무 심해져서 수렴이 느려져요."
>
> "그리고 핵심은 **매 epoch마다 같은 사진이 다른 변형으로 보인다**는 거예요. 예를 들어 `Apple_rust/image (5).JPG`가 epoch 1에선 좌우 반전 + 8도 회전, epoch 2에선 그대로 + -3도 회전, epoch 5에선 다시 좌우 반전 + 14도 회전, 이런 식으로 매번 다르게 들어가요. 디스크엔 원본 1장이지만 모델 입장에선 사실상 25 epoch × 다른 변형 = 25장으로 보여요. 무한 데이터 효과가 나는 거예요."
>
> "그리고 **val에는 augmentation 안 걸어요**. resize랑 normalize만. 정직한 측정을 위해서요."

### 7-3. 모델 구조 — ScratchCNN

`src/leaffliction/models/scratch_cnn.py` 띄우며:

🎤

> "모델은 두 부분이에요. `self.features`가 conv block 4개 쌓은 특징 추출기고, `self.head`가 GAP + Dropout + Linear로 된 분류기예요."
>
> "한 conv block은 7층이에요. `Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2)` 순서고요."

**Conv2d 설명**:

> "`Conv2d`는 3×3 짜리 학습 가능한 필터로 이미지를 훑어서 새 채널을 만드는 거예요. 채널 개념이 좀 추상적인데요, 채널 1개 = 같은 사진을 한 가지 관점에서 본 흑백 지도라고 생각하시면 돼요. 입력은 RGB라 3 채널이고, 첫 conv block 지나면 32 채널, 다음 64, 128, 마지막은 256 채널. 각 채널이 '다른 종류의 패턴이 사진의 어디에 있나'를 보여주는 지도가 되고, 필터 내용은 학습으로 자동 결정돼요."

**BatchNorm / ReLU / MaxPool**:

> "BatchNorm은 한 layer의 출력을 평균 0, 분산 1로 정규화하는 거예요. 학습이 안정적이 되고요, 입력 사진의 밝기나 대비가 좀 달라져도 모델이 일관되게 학습해요."
>
> "ReLU는 `max(0, x)` 함수예요. 음수는 0으로 만들고 양수는 그대로 통과. 신경망에 비선형성을 주는 핵심이에요. ReLU가 없으면 layer를 아무리 쌓아도 결국 직선 하나로 표현되는 선형 모델이 돼버려요."
>
> "MaxPool은 2×2 영역에서 최대값만 남기는 거예요. 공간 해상도가 절반으로 줄어요. 256 → 128 → 64 → 32 → 16. 작은 디테일은 버리고 큰 그림에 집중하게 만드는 거예요."
>
> "4 block 거치면 입력 `(3, 256, 256)`이 `(256, 16, 16)`이 돼요. 채널이 늘어나면서 공간이 줄어드는 깔때기 구조 — CNN의 전형이에요."

**Head**:

> "head는 4층이에요. `AdaptiveAvgPool2d(1)`이 GAP, Global Average Pooling이고요. 마지막 conv 출력 `(256, 16, 16)`에서 위치 정보를 평균 내서 `(256,)` 벡터로 압축해요. '사진 전체에 256개 패턴이 얼마나 강한가'라는 정보로 줄이는 거예요. 옛날 FC layer가 너무 무거웠던 문제를 해결하고 위치 invariance까지 얻어요."
>
> "Dropout(0.4)는 학습 중에 40% 뉴런을 무작위로 끄는 거예요. 특정 뉴런이 답을 외우는 걸 막아서 overfitting을 완화해요. 평가할 땐 dropout 자동으로 꺼지고요."
>
> "마지막 Linear(256, 8)이 8 클래스 logit을 만들어요. softmax 씌우면 확률이 되고요. 총 파라미터는 약 1.18M개입니다."

### 7-4. 학습 루프 — Forward, Backward, 역전파

`src/leaffliction/trainer.py` 보여주며:

🎤

> "학습 루프의 핵심은 PyTorch 5줄이에요."

```python
logits = model(x)              # ① forward
loss = criterion(logits, y)    # ② loss
optimizer.zero_grad()          # ③ 이전 gradient 지움
loss.backward()                # ④ backward (역전파)
optimizer.step()               # ⑤ weight 업데이트
```

> "한 줄씩 보면, **forward는 현재 weight로 예측을 만드는 단계**예요. 입력 사진이 conv block 4개 거쳐서 head까지 가면 8개 클래스에 대한 logit이 나와요."
>
> "**loss**는 예측이랑 정답이 얼마나 다른지 점수 매기는 거예요. `CrossEntropyLoss`를 쓰는데, 정답 클래스 확률이 1에 가까우면 loss가 0에 가깝고, 멀면 loss가 커져요."
>
> "**backward, 이게 바로 역전파, backpropagation이에요**. 직관적으로 설명하면, 모델 안에는 수백만 개의 weight가 있는데, 이 weight 하나하나가 loss에 얼마나 영향을 줬는지 알아야 weight를 어떻게 조정할지 알 수 있잖아요. 그게 `∂loss/∂weight` — 미분값이고요."
>
> "근데 신경망은 합성 함수예요. `loss = L(head(conv4(conv3(conv2(conv1(x))))))`. 가장 안쪽 `conv1`의 weight가 loss에 미치는 영향을 직접 미분할 수 없어요. **chain rule, 연쇄법칙**을 layer 끝에서 시작 방향으로 거꾸로 적용해야 해요. 그래서 '역'전파라고 해요. forward는 입력 → 출력 방향, backward는 loss에서 시작해서 입력 쪽으로 도함수를 곱해가는 방향."
>
> "PyTorch의 **autograd**가 이걸 자동으로 해줘요. forward 중에 computation graph를 기억해뒀다가 `loss.backward()` 한 줄 호출하면 chain rule이 자동 적용돼서 모든 weight의 gradient가 계산돼요. 1980년대엔 이걸 손으로 미분식 작성했다는데, autograd가 deep learning이 폭발한 결정적 이유 중 하나예요."
>
> "마지막 **optimizer.step()**은 gradient의 반대 방향으로 weight를 살짝 옮기는 거예요. `weight = weight - learning_rate × gradient` 이런 식으로요. gradient descent라고 부르고요."
>
> "이 5줄을 batch마다 한 번씩 돌리고, batch는 32장씩 묶어서 train 5,783장이면 한 epoch에 약 181 step이에요. 25 epoch면 총 약 4,500 step. 그동안 weight가 4,500번 살짝씩 정답 쪽으로 옮겨가면서 학습되는 거예요."

### 7-5. Loss / Optimizer / Scheduler / Early stop — 그리고 epoch=25 이유

🎤

> "Loss는 방금 말씀드린 `CrossEntropyLoss`고요."
>
> "Optimizer는 `Adam(lr=1e-3, weight_decay=1e-4)`. Adam은 weight마다 적정 step 크기를 적응적으로 조절하는 알고리즘이에요. momentum + RMSprop을 결합한 거고, plain SGD보다 수렴이 빨라요. `weight_decay`는 L2 regularization이라고도 부르는데, weight가 너무 커지는 걸 막아서 overfitting을 완화해요."
>
> "Scheduler는 `ReduceLROnPlateau(factor=0.5, patience=2)`. val_accuracy가 2 epoch 연속 정체되면 learning rate를 절반으로 줄여요. 학습 후반에 미세 조정할 때 유용해요."
>
> "Early stopping은 `patience=5`로 설정했어요. val_acc가 5 epoch 동안 0.1% 이상 개선 안 되면 자동으로 학습 중단. overfitting 진입 직전에 끊어주는 안전장치예요."

epoch=25 질문 들어오면:

> "`--epochs 25`는 상한선이에요. 실제로 25 epoch을 다 도는 경우는 거의 없고, 보통 17~22 epoch 부근에서 early stopping이 발동돼요."
>
> "왜 그러냐면, 이 데이터셋이 명확해서 학습이 빨리 수렴해요. 통제된 회색 배경에 잎 1장만 있는 데이터라 클래스 경계가 뚜렷하거든요. 보통 epoch 14~17 정도에 best val_acc 99.8%대를 찍고, 그 이후엔 0.1% 이상의 개선이 안 나와요. patience=5라서 best 이후 5 epoch 정체 누적되면 종료. 그래서 19~22에서 자연스럽게 끝나요."
>
> "25라는 숫자 자체는 — 너무 적으면(예: 10) underfitting 위험이 있고, 너무 많으면(예: 100) early stopping 안 걸렸을 때 시간 낭비예요. 비슷한 명확한 분류 문제에서 20 epoch 전후 수렴이 ML 경험칙이라, 그 위에 약간 마진을 둔 값이에요."

> "참고로 `WeightedRandomSampler(replacement=True)` 때문에 한 epoch에 모든 사진이 다 보장되진 않아요. Apple_healthy 같은 큰 클래스는 한 epoch에 약 45% 사진이 등장 안 할 수도 있어요. 하지만 25 epoch 누적이면 확률적으로 거의 100% 모든 사진이 보입니다. 그리고 매번 다른 random 변형으로 보이니까 사실상 사진별 14~80회 노출이고, 학습량은 충분히 누적돼요."

### 7-6. 결과 시각화

`artifacts/learning_curves.png` 띄우며:

🎤

> "학습 곡선이에요. train_loss랑 val_loss가 함께 감소하면서 수렴해요. 보시면 어떤 epoch에선 train_loss가 val_loss보다 높은 경우도 있는데, 이게 좀 특이해 보이지만 사실 정상이에요. train batch에만 augmentation을 걸기 때문에 train이 val보다 더 어려운 문제거든요. **이건 overfitting의 정반대 신호** — augmentation이 잘 작동한다는 증거예요."

`artifacts/confusion_matrix.png` 띄우며:

> "혼동 행렬이에요. 대각선이 거의 다 채워져 있고 비대각선은 1~2장 수준. 8 클래스 중 4개가 100%, 나머지도 99%대고요."

`artifacts/classification_report.txt`와 `artifacts/metadata.json` 보여주며:

> "scikit-learn classification report는 클래스별 precision, recall, f1을 보여주고요. `metadata.json`엔 best_epoch, val_accuracy, class layout이 들어있어요. 이 파일이 `trained_models.zip`에 같이 들어가서 `predict.py`가 클래스 라벨 매핑하는 데 써요."

### 7-7. 데이터 누수 회피 — 가장 자주 묻는 질문

🎤

> "정확도가 99.8%다 보니까 'overfitting 아니냐, 누수 있는 거 아니냐' 의심하실 수 있어요. 핵심은 **augmentation이랑 split의 순서**예요."
>
> "잘못된 순서가 augment-then-split이에요. `augmented_directory/`처럼 augmented된 디스크 데이터셋을 그대로 80/20 split하면, 같은 원본의 변형들 — `image (1).JPG`랑 `image (1)_Flip_0.JPG` — 이 train이랑 val에 흩어져요. 그럼 모델이 train에서 본 사진의 거의 같은 변형을 val에서 만나서 가짜 100% 정확도가 나와요. 이걸 데이터 누수라고 해요."
>
> "저희 순서는 split-then-augment예요. **원본 `images/`만 split해놓고**, augmentation은 train batch에 메모리에서만 적용해요. 변형이 디스크에 안 남으니까 val에 노출될 경로 자체가 없어요. 누수 위험 0."
>
> "실제로 v1을 augmented_directory로 학습했을 때 100% 나왔어요. 의심스러웠고요. v2로 원본 + online augmentation으로 바꾸니까 99.79%로 자연스럽게 떨어졌어요. 이게 정직한 결과예요."

Transfer 비교:

> "그리고 비교용으로 EfficientNet-B0 transfer learning도 같이 학습할 수 있게 했어요. EfficientNet-B0은 ImageNet 100만 장으로 사전학습된 모델이에요. CNN 초기 layer가 학습한 '선, 곡선, 텍스처' 인식 능력은 잎 사진에도 그대로 활용 가능하거든요. 마지막 1000-class 분류층만 8-class로 갈아끼우고 두 단계로 fine-tune해요. epoch 1~5은 backbone 동결하고 classifier만 학습, epoch 6부터 전체 unfreeze하고 LR을 1/10로 줄여서 미세조정. 결과는 99.86%로 ScratchCNN과 거의 같아요. 두 모델이 비슷한 결과를 낸다는 게 데이터셋이 명확하다는 또 다른 증거예요."

### 📂 관련 코드 (이 섹션 전체)
- `src/leaffliction/models/scratch_cnn.py` — ScratchCNN. 4 conv block + GAP head.
- `src/leaffliction/models/transfer.py` — EfficientNet-B0 wrapper. `freeze()` / `unfreeze()` 메서드.
- `src/leaffliction/trainer.py` — `train()` 함수. PyTorch 5줄 루프, optimizer, scheduler, early stop, two-stage fine-tune 분기.
- `src/leaffliction/dataset.py::LeafDataset` — PyTorch Dataset. `__getitem__`에서 transform 매번 새로 적용 → online augmentation.
- `train.py` — typer CLI. stratified split, WeightedRandomSampler 구성, 모델 dispatch, 결과 시각화 + zip + signature.

---

## 8. Part 4 (3/4) — Unit_test1 (Apple)

### 🎯 평가자가 확인하는 것
`test_images/Unit_test1/`의 10장에 대해 맞춘 개수만큼 점수. 파일명을 무작위로 바꿔서 부정 방지.

### 💻 명령
```bash
# 폴더 통째로 — 다중 모드, 자동 self-check
./predict.py /tmp/test_images/Unit_test1/

# PNG도 일괄 저장하고 싶으면
./predict.py /tmp/test_images/Unit_test1/ --save /tmp/out_unit1/

# 단일 이미지 — PDF 예시 그대로
./predict.py /tmp/test_images/Unit_test1/Apple_healthy1.JPG
```

### 🎤 대사

> "Unit_test1은 Apple 4 클래스에서 뽑은 10장이에요. `predict.py`에 폴더를 통째로 넘기면 자동으로 안의 `*.JPG`를 다 수집해서 한 번에 예측해요. 단일 이미지로 호출하면 PDF 예시처럼 matplotlib figure를 띄우고, 폴더면 콘솔 표로 간결하게 출력해요."

출력 예시:

```
Predicting 10 images with model=scratch...
  OK   Apple_healthy      (99.8%)  ← Apple_healthy1.JPG
  OK   Apple_Black_rot    (99.1%)  ← Apple_BlackRot2.JPG
  ...
Self-check: 10/10 = 100.00%
```

> "파일명이 클래스명으로 시작하면 — 평가표 예시처럼 `Apple_healthy1.JPG` — 자동 self-check이 동작해요. 평가자가 파일명을 무작위로 바꾸면 self-check은 그냥 skip하고 빈 칸으로 표시되지만 예측은 그대로 동작해요. **파일명은 예측 입력에 안 쓰여요**, self-check 보조 정보일 뿐이에요. 모델은 zip 안의 metadata.json에서 클래스 라벨을 읽고, 사진 픽셀만 보고 예측해요."

확률 100%가 의심스럽다고 하면:

> "참고로 신뢰도가 100.0%로 뜨는 건 softmax 출력값이라서요, 진짜 확률이 100%라는 보장은 아니에요. 모델이 매우 확신했다는 의미고요, 신경망은 학습이 잘 되면 과잉 자신감을 가지는 경향이 있어요. softmax exponential 때문에 logit 차이가 좀 작아도 1.0에 거의 붙어버려요. 일반 야외 사진처럼 더 도전적인 입력이면 70~95%로 떨어집니다."

### 📂 관련 코드
- `predict.py` — typer CLI. 인자가 단일 파일이면 figure 모드, 다중 / 디렉토리면 콘솔 표 모드 자동 dispatch. `_guess_class_from_name`이 파일명 prefix 매칭으로 self-check.
- `src/leaffliction/predictor.py`
  - `load_artifact(zip, prefer)` — zip 한 번 unzip 후 모델 + classes 로드. 다중 모드에서 모델 1회만 load.
  - `predict_one(artifact, image)` — PIL load → resize → normalize → forward → softmax argmax.
  - `render(result, save)` — 2-패널 figure (원본 + mask).

---

## 9. Part 4 (4/4) — Unit_test2 (Grape)

### 🎯 평가자가 확인하는 것
Grape 4 클래스에서 뽑은 10장. **10장 다 틀리면 누수 의심하라고 평가표가 명시**.

### 💻 명령
```bash
./predict.py /tmp/test_images/Unit_test2/
./predict.py /tmp/test_images/Unit_test2/ --save /tmp/out_unit2/
```

### 🎤 대사

> "Unit_test2는 Grape 쪽 10장이에요. 같은 폴더 다중 모드로 한 번에 돌릴게요."
>
> "평가표가 'Unit_test2 다 틀리면 학생이 어떻게 validation 정확도를 높였는지 의심하라'고 명시해놨어요. 누수 가능성을 짚는 거죠. 저희는 §7-7에서 설명드렸듯이 split-then-augment 순서를 지켜서 누수를 차단했고, Unit_test2가 통과한다는 게 그 정당성의 또 다른 증거예요."

### ⚠️ 함정
10장 다 틀리면 누수 의심 강조. §7-7 누수 회피 설명 다시 한 번.

---

## 10. 자주 받는 질문 (Q&A)

### Q1. CNN을 왜 직접 만들었어요? Transfer learning이 더 쉽지 않아요?
> "둘 다 만들었어요. Default는 ScratchCNN인데요, 이유는 두 가지예요. 첫째, 모든 layer를 제가 직접 설계해서 한 줄씩 설명할 수 있다는 게 디펜스에 안전하고요. 둘째, 평가표가 '모델 설명' 항목에 5점을 배정했기 때문에 black-box pretrained 모델보다 직접 만든 모델이 설명 면에서 유리해요. Transfer는 비교용 + 실제 production에선 어떻게 쓰는지 보여주는 용도로 같이 뒀어요."

### Q2. EfficientNet-B0은 왜 선택했어요?
> "비슷한 정확도에서 파라미터/연산량 효율이 좋아서 modern 표준 baseline이에요. ResNet-50이 25M 파라미터면 B0은 5M으로 1/5 수준이고, CPU에서도 추론이 빠릅니다."

### Q3. 데이터 누수가 정확히 뭐예요?
> "모델이 평가 단계에서 보면 안 되는 정보를 학습 중에 이미 본 상태예요. 가장 흔한 게 augment-then-split이에요. 같은 원본의 변형들이 train이랑 val에 흩어지면 모델이 train에서 외운 답을 val에서 거의 그대로 만나서 가짜 100% 정확도가 나와요. 저희는 원본만 split하고 augmentation은 train batch 메모리에만 적용해서 차단했어요."

### Q4. WeightedRandomSampler 안 쓰고 그냥 augmented_directory로 균형 맞추면 안 돼요?
> "augmented_directory는 PDF 요구사항 때문에 만들지만 학습엔 안 써요(누수 때문). 그래서 클래스 불균형 보정할 다른 방법이 필요했고 그게 WeightedRandomSampler예요. 디스크 저장 없이 batch 단위에서 같은 효과 + 누수 회피."

### Q5. seed=42는 왜 42예요?
> "숫자 자체는 의미 없어요. ML 커뮤니티 관용 기본값이라 그냥 썼고요, 핵심은 고정값이라는 것 자체예요. 0이든 12345든 다 됩니다. 같은 seed면 train.py랑 eval_val.py가 정확히 같은 val 1,445장을 뽑게 돼서 재현성이 보장되는 게 중요한 거예요."

### Q6. 정확도 99.8%면 너무 좋은데 진짜인가요?
> "세 가지 증거를 드릴 수 있어요. 첫째, 이 데이터셋이 통제된 회색 배경에 잎 1장씩 찍힌 학습용 데이터라 클래스 시각 차이가 명확해서 모델이 풀기 쉬운 문제예요. 둘째, ScratchCNN이랑 EfficientNet 두 다른 방식이 99.79%, 99.86%로 0.07pp 이내라서 한쪽 모델의 트릭이 아니라 데이터셋 명확성의 증거예요. 셋째, confusion matrix가 자연스러워서 1~2장 misclassified가 클래스별로 흩어져 있어요. 추가로 `eval_val.py`로 즉시 재검증 가능합니다."

### Q7. Overfitting은 어떻게 막았어요?
> "다섯 가지를 동시에 써요. Dropout 0.4, weight_decay 1e-4, online augmentation, early stopping patience=5, ReduceLROnPlateau. learning_curves.png에서 train이랑 val이 함께 수렴하는 게 증거고요, 종종 train_loss > val_loss인 epoch도 있는데 그건 train batch만 augmented라서 train이 더 어려운 거예요."

### Q8. epoch=25는 왜 25예요?
> "상한선이에요. 명확한 분류 문제에서 보통 20 epoch 전후로 수렴하는 게 ML 경험칙이거든요. patience=5 여유까지 더해서 25로 잡았어요. 실제로는 17~22 epoch에서 early stopping 발동돼서 다 안 채워요."

### Q9. forward / backward가 뭐예요?
> "forward는 현재 weight로 예측을 만드는 단계예요. backward는 backpropagation, 역전파라고 부르고요. loss가 weight 하나하나에 얼마나 영향을 줬는지 chain rule로 계산하는 거예요. layer가 합성 함수라 끝(loss)에서 시작(입력) 방향으로 거꾸로 미분을 전파해야 하는데, PyTorch의 autograd가 이걸 자동으로 해줘요. 우리는 forward만 정의하면 `loss.backward()` 한 줄로 모든 weight의 gradient가 계산됩니다."

### Q10. uv는 왜 쓰나요?
> "pip + virtualenv + pyenv + pip-tools를 통합한 modern Python 도구예요. Rust로 작성됐고 pip 대비 10~100배 빠르고요, `uv.lock`으로 의존성 재현성을 보장해요."

### Q11. 모르는 질문이 나오면?
> "솔직히 '그건 모르겠는데 코드 보면서 같이 확인하겠습니다' 하고 코드 띄움. 정직성이 점수에 더 안전해요. 평가표 0점 조건은 '설명을 못 함'이지 '한두 개 모름'이 아니에요."

---

## 11. 위험 시나리오 대응

### `make verify` 실패 (signature 불일치)
```bash
ls -la trained_models.zip augmented_directory.zip
shasum trained_models.zip augmented_directory.zip
# → 백업 USB로 재시도. 최후엔 ./train.py images/ --epochs 25 재실행 (~50분)
```

### `uv sync` 실패 (plantcv 빌드 에러)
```bash
uv pip install plantcv --no-build-isolation
# 또는
python -m pip install -e .
```

### matplotlib 창이 안 뜨는 환경 (SSH/Docker)
```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png
```

### Unit_test에서 다 틀리는 경우
§7-7 누수 회피 설명 한 번 더 + `artifacts/confusion_matrix.png`로 내부 분포에선 잘 동작한다는 증거. 학습 분포 밖(예: 야외 휴대폰 사진)에선 자연스럽게 정확도 떨어질 수 있음을 인정.

### `./Distribution.py: command not found`
```bash
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py
# 또는 venv 활성화 확인
source .venv/bin/activate
# 또는 uv 직접
uv run python Distribution.py ./images
```

### 평가자가 다른 dataset path를 줌
모든 entrypoint가 path 인자를 받아서 우리 `images/`에 종속 안 됨.
```bash
./Distribution.py /평가자/경로
./predict.py "/평가자/Unit_test1/"
```

---

## 12. 마무리 한 줄

> "PDF 5개 entrypoint와 평가표의 0점 함정(augmented_directory 균형, signature 일치, 100+ 이미지 ≥ 90%) 다 통과 + 데이터 누수 회피한 정직한 결과입니다. ScratchCNN을 직접 설계해서 모든 layer를 설명할 수 있고, EfficientNet-B0 transfer model도 비교용으로 두어 CNN의 production 활용 방식까지 함께 보여드릴 수 있습니다."

---

## 📚 부록: 코드 흐름 상세 워크스루

이 부록은 디펜스 당일 읽는 게 아니라 **사전에 코드를 이해하려고 공부할 때** 보는 용도예요. 각 Part가 어떻게 동작하는지 시간 순서대로 따라가면서, 나중에 코드를 다시 보면 "아 이게 이거구나" 하고 알 수 있게 풀어 썼어요.

### 전체 파일 지도 (한 번 더, 부록 시작 전에)

```
사용자가 명령 입력 (예: ./Distribution.py ./images)
         ↓
루트 entrypoint (.py 파일) — typer가 CLI 인자 파싱
         ↓
src/leaffliction/<해당 모듈>.py — 실제 로직
         ↓
matplotlib 화면 표시 또는 디스크에 파일 저장
```

루트의 5개 entrypoint 파일은 다 **얇은 wrapper**예요. 진짜 일은 `src/leaffliction/` 안의 모듈들이 합니다.

---

### A. Distribution.py 흐름

#### A.1 파일 지도

```
Distribution.py
    │  ① typer로 CLI 파싱 (directory, --save)
    │  ② die() — 에러 시 정중하게 종료
    │
    ├──→ dataset.py::discover_classes(directory)
    │       └─ 폴더 구조 보고 {클래스명: [사진 경로 list]} 반환
    │
    └──→ viz.py::pie_and_bar(counts, title, save)
            └─ matplotlib으로 좌우 두 패널 그림
```

#### A.2 사용자가 명령을 입력한 순간부터 차트가 뜨기까지

```
사용자: ./Distribution.py ./images
   ↓
[1] typer가 "./images" 문자열을 Path 객체로 변환
   ↓
[2] main() 함수 호출:
       directory = Path("./images")
       save = None
   ↓
[3] discover_classes(directory) 호출
       ↓
       ./images 안에 사진이 직접 있나? → 없음 (자식 폴더만 있음)
       ↓
       자식 폴더 순회: Apple_Black_rot, Apple_healthy, Apple_rust, ...
       ↓
       각 자식 폴더 안의 *.jpg 모음 (sorted)
       ↓
       반환: {
         "Apple_Black_rot": [Path("images/Apple_Black_rot/image (1).JPG"), ...],
         "Apple_healthy":   [Path("images/Apple_healthy/image (1).JPG"), ...],
         ... (8 클래스)
       }
   ↓
[4] main이 받은 dict로 카운트 계산:
       counts = {name: len(paths) for name, paths in classes.items()}
              = {"Apple_Black_rot": 621, "Apple_healthy": 1640, ...}
   ↓
[5] pie_and_bar(counts, title="images", save=None) 호출
       ↓
       fig, (ax_pie, ax_bar) = plt.subplots(1, 2)
       ↓
       ax_pie.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
       ax_bar.bar(counts.keys(), counts.values())
       ↓
       막대 위에 ax_bar.text(...)로 정확한 숫자 표시
       ↓
       plt.show() ← matplotlib 창 뜸
```

#### A.3 `discover_classes`가 똑똑한 이유

```python
def discover_classes(root):
    direct_images = _images_in(root)
    if direct_images:
        return {root.name: direct_images}      # Layout 1
    
    classes = {}
    for child in sorted(...):                  # Layout 2 (우리 표준)
        child_images = _images_in(child)
        if child_images:
            classes[child.name] = child_images
            continue
        for grand in sorted(...):              # Layout 3 (중첩 구조)
            ...
```

3가지 폴더 구조를 다 처리:

```
Layout 1 (단일 폴더 — Apple_healthy/ 만 넘긴 경우):
  Apple_healthy/
    ├── image (1).JPG
    ├── image (2).JPG
    └── ...
  → {"Apple_healthy": [Path, Path, ...]}

Layout 2 (우리 표준 — images/ 같은 구조):
  images/
    ├── Apple_healthy/
    │   ├── image (1).JPG
    │   └── ...
    ├── Apple_Black_rot/
    │   └── ...
    └── ... (8 클래스)
  → {"Apple_healthy": [...], "Apple_Black_rot": [...], ...}

Layout 3 (group으로 한 번 더 묶인 경우 — Apple/ 안에 healthy/, scab/...):
  Apple/
    ├── apple_healthy/
    └── apple_scab/
  → {"apple_healthy": [...], "apple_scab": [...]}
```

평가자가 어떤 구조로 데이터셋을 주든 동작하게 만든 거예요.

**`sorted()`가 두 군데 들어간 이유**: 같은 폴더 구조면 항상 같은 순서로 클래스가 매핑돼서, train.py가 학습할 때 본 클래스 인덱스(0=Apple_Black_rot, 1=Apple_healthy, ...)와 eval_val.py가 보는 인덱스가 일치하게 보장돼요. **재현성의 출발점**.

---

### B. Augmentation.py 흐름

#### B.1 파일 지도

```
Augmentation.py
    │  ① target.is_file() ─→ 단일 모드
    │  ② target.is_dir()  ─→ 배치 모드
    │
    ├─→ [단일] augment.py::apply_op + save_with_suffix
    │       └─ 6 op 다 적용해서 sibling 파일 저장
    │       그리고 viz.py::grid로 화면 표시
    │
    └─→ [배치] augment.py::balance_directory
            └─ 클래스마다 augment해서 target_count까지 채움
            └─ zip_directory로 .zip 자동 생성
```

#### B.2 단일 모드 (PDF 시연용)

```
사용자: ./Augmentation.py "images/Apple_healthy/image (1).JPG"
   ↓
[1] target = Path("images/Apple_healthy/image (1).JPG")
       target.is_file() → True → 단일 모드 분기
   ↓
[2] rgb = load_image(target)
       └─ PIL.Image.open(target).convert("RGB") → numpy array (H, W, 3)
   ↓
[3] outputs = [("Original", rgb)]  # grid 표시용
   ↓
[4] 6 op 순회:
   for name in AUGMENTATION_OPS:                # ["Flip", "Rotate", "Skew", ...]
       aug = apply_op(name, rgb)                # 변형된 numpy array
       save_with_suffix(target, aug, name)      # image (1)_Flip.JPG로 저장
       outputs.append((name, aug))              # grid에도 추가
   ↓
[5] grid(outputs)
       └─ matplotlib subplots 1×7 (원본 1 + 변형 6)
       └─ 각 패널에 ax.imshow(img) + set_title(label)
```

`AUGMENTATION_OPS`가 dict라서 순회 순서가 Python 3.7+의 삽입 순서(Flip → Rotate → Skew → Shear → Crop → Distortion)로 결정적이에요. 매번 호출해도 같은 순서.

#### B.3 배치 모드 (Part 1 verify용)

```
사용자: ./Augmentation.py images/
   ↓
[1] target.is_dir() → True → 배치 모드
   ↓
[2] balance_directory(target, output="augmented_directory", ...)
       ↓
       [a] discover_classes(src)로 클래스별 사진 list 수집
       [b] target = max(클래스별 길이) = 1640 (Apple_healthy)
       ↓
       [c] 각 클래스에 대해:
           for cls, paths in classes.items():
               # 원본 복사
               for src in paths[:target]:
                   shutil.copy2(src, dst/<cls>/<filename>)
               produced = 복사한 갯수
               
               # 부족분만큼 augmentation으로 채움
               while produced < 1640:
                   base = rng.choice(paths)             # 같은 클래스 내 random
                   img = load_image(base)
                   op_name, aug = apply_random_op(img)  # 6 op 중 random 1개
                   save: <stem>_<op>_<counter>.JPG
                   produced += 1
       ↓
       [d] 끝나면 zip_directory(dst) 자동 호출
           └─ zipfile.ZIP_DEFLATED로 augmented_directory.zip 생성
```

처리되는 클래스별 결과 시각화:

```
원본 (불균형)              결과 (균등)
─────────────             ─────────────
Apple_healthy   1640      Apple_healthy   1640  (그대로 복사)
Apple_Black_rot  621      Apple_Black_rot 1640  (621 복사 + 1019 augment)
Apple_rust       275      Apple_rust      1640  (275 복사 + 1365 augment)
Apple_scab       630      Apple_scab      1640  (630 복사 + 1010 augment)
... (8 클래스)            ... (8 × 1640 = 13,120장)
```

#### B.4 `apply_op`의 한 줄짜리 핵심

```python
def apply_op(name, image):
    transform = AUGMENTATION_OPS[name]
    return transform(image=image)["image"]
```

Albumentations API는 `transform(image=array)`를 호출하면 `{"image": 변형된_array}` dict를 반환해요. `["image"]`로 키만 추출. 함수 본체는 사실 한 줄. dict 자료구조 덕분에 6 op을 dict + 한 줄 호출로 표현 가능.

---

### C. Transformation.py 흐름

#### C.1 파일 지도

```
Transformation.py
    │  ① image 인자 → 단일 모드 (한 figure에 6 변환 + hist)
    │  ② -src/-dst 인자 → 배치 모드 (디렉토리 통째로 처리)
    │
    └─→ transform.py::
            load_rgb         (PIL → numpy)
            _binary_mask     (5단계 mask 파이프라인 — 핵심)
            all_transforms   (6 변환 dict 반환)
            color_histogram  (9 채널 hist)
```

#### C.2 단일 모드 흐름

```
사용자: ./Transformation.py "images/Apple_healthy/image (1).JPG"
   ↓
[1] rgb = load_rgb(image)             # (H, W, 3) numpy uint8 array
   ↓
[2] outs = all_transforms(rgb)        # 6 변환 한 번에
       반환: {"Original": ..., "GaussianBlur": ..., "Mask": ...,
              "RoiObjects": ..., "AnalyzeObject": ..., "Pseudolandmarks": ...}
   ↓
[3] hist = color_histogram(rgb)       # 9 채널 히스토그램
       반환: {"red": array(256,), "green": ..., ..., "blue-yellow": ...}
   ↓
[4] matplotlib figure 구성:
       
       fig = plt.figure(figsize=(16, 9))
       
       ┌─────┬─────┬─────┬─────┬─────┬─────┐
       │ Org │ Blur│ Mask│ ROI │Analy│ Land│   ← subplot2grid((2,6), (0, i))
       ├─────┴─────┴─────┴─────┴─────┴─────┤
       │       9채널 color histogram         │   ← subplot2grid((2,6), (1,0), colspan=6)
       └───────────────────────────────────┘
   ↓
[5] 각 윗줄 셀: ax.imshow(img) (2D는 cmap="gray", vmin=0, vmax=255 강제)
[6] 아랫줄: 9 채널 plot 모두 한 axes에 + legend
[7] plt.show()
```

`vmin=0, vmax=255` 고정이 왜 중요한지: matplotlib이 흰 픽셀이 많은 mask 같은 이미지를 auto-normalize하면 어두운 회색으로 보일 수 있어요. 명시적으로 0~255 범위 강제로 그 버그 차단.

#### C.3 `_binary_mask` 5단계 파이프라인 시각화

```
입력: rgb (H, W, 3)  ─ healthy 사과 잎 + 회색 배경
   ↓
[1] cv2.cvtColor(rgb, RGB2LAB) → lab (H, W, 3)
       a = lab[..., 1] - 128       ← 녹↔적 축, 회색은 0 근처
       b = lab[..., 2] - 128       ← 청↔황 축, 회색은 0 근처
       chroma = sqrt(a² + b²)       ← 0~255 (회색=0, 녹/갈색=큼)
   ↓
   chroma 시각화 (회색=어두움, 색 있는 부분=밝음):
   ┌─────────────┐
   │ ░░░░░░░░░░░ │   ← 회색 배경 (chroma ≈ 0)
   │ ░░██████░░░ │   ← 잎 (chroma ≈ 100)
   │ ░░██▓▓██░░░ │   ← 갈색 병변 (chroma ≈ 80)
   │ ░░░░░░░░░░░ │
   └─────────────┘
   ↓
[2] cv2.threshold(chroma, 0, 255, BINARY+OTSU)
       Otsu가 자동으로 cutoff(예: 30) 잡고 binary 변환
   ↓
   binary:
   ┌─────────────┐
   │ ░░░░░░░░░░░ │   ← 0 (배경)
   │ ░░████████░ │   ← 255 (잎)
   │ ░░████████░ │
   │ ░░░░░░░░░░░ │
   └─────────────┘
   ↓
[3] cv2.morphologyEx(binary, MORPH_OPEN, kernel=3×3)
       (erosion + dilation) — 작은 노이즈 점 제거
   ↓
[4] cv2.connectedComponentsWithStats
       binary 안의 모든 "흰 덩어리"를 찾아서 가장 큰 것만 유지
       → 잎 외 다른 작은 잡티 제거
   ↓
[5] scipy.binary_fill_holes
       잎 외곽선은 잡혔지만 안쪽에 작은 검은 구멍들이 있음
       → 안쪽을 다 흰색으로 채움
   ↓
[6] pcv.fill(bin_img=filled, size=200)
       size 200 이하의 잡티 한 번 더 제거 (안전 마진)
   ↓
출력: binary mask (H, W) ─ 잎 부분 255, 배경 0
```

이 mask가 `gaussian_blur`, `mask`, `roi_objects`, `analyze_object`, `pseudolandmarks` **5개 함수의 공통 전처리**. mask 한 번 잘 잡으면 나머지가 다 깔끔하게 됨.

#### C.4 9채널 histogram이 만들어지는 과정

```python
def color_histogram(rgb):
    hsv = cv2.cvtColor(rgb, RGB2HSV)       # 색공간 변환 1
    lab = cv2.cvtColor(rgb, RGB2LAB)       # 색공간 변환 2
    
    channels = {                            # 9 채널 dict로 묶음
        "blue":  rgb[..., 2], "green":      rgb[..., 1], "red":            rgb[..., 0],
        "hue":   hsv[..., 0], "saturation": hsv[..., 1], "value":          hsv[..., 2],
        "lightness": lab[..., 0], "green-magenta": lab[..., 1], "blue-yellow": lab[..., 2],
    }
    
    out = {}
    total = H * W                           # 총 픽셀 수
    for name, ch in channels.items():
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        out[name] = 100.0 * hist / total    # 비율(%)로 정규화 ← 사진 크기 무관 비교 가능
    return out
```

데이터 흐름:

```
rgb (H, W, 3)                   ── 한 사진
     ↓
[색공간 변환]
RGB (직접 사용) ─── R, G, B 각각 (H, W) 채널 3개
HSV (변환)     ─── H, S, V 각각 (H, W) 채널 3개
LAB (변환)     ─── L, a*, b* 각각 (H, W) 채널 3개
     ↓ (9개 (H,W) 배열)
[각 채널마다 np.histogram(bins=256)]
     ↓
9개의 길이 256 배열 (각 픽셀 값의 빈도)
     ↓
[픽셀 수로 나눠서 %화]
     ↓
{"red": array(256,), "green": ..., ..., "blue-yellow": ...}
```

---

### D. train.py 흐름 (가장 복잡, 가장 중요)

#### D.1 전체 흐름도

```
사용자: ./train.py images/ --epochs 25
   ↓
[Phase 1: 준비]
   set_seed(42) ──── 모든 RNG 고정 (Python/NumPy/PyTorch/cuDNN)
   ↓
   _build_loaders() ── train/val DataLoader 만들기
       │
       ├─ LeafDataset(images, train_tf) ─── 5,783장
       ├─ LeafDataset(images, val_tf)   ─── 1,445장 (같은 폴더, 다른 transform)
       ├─ train_test_split(stratify, seed=42) ─── 80/20 인덱스 분할
       ├─ WeightedRandomSampler ─── weight = 1/클래스 크기
       └─ DataLoader (num_workers=8, persistent_workers=True)
   ↓
[Phase 2: 학습 — 25 epoch 상한, early stop으로 자동 종료]
   _train_one("scratch", ScratchCNN(), train_loader, val_loader, TrainConfig)
       │
       └─ trainer.py::train()
              │
              for epoch in 1..25:
                  ├─ _epoch(model, train_loader, criterion, optimizer)  ← 학습
                  │      └─ batch마다 forward/loss/backward/step
                  ├─ _epoch(model, val_loader, criterion, None)          ← 평가 (gradient 없음)
                  ├─ scheduler.step(val_acc)
                  ├─ best 갱신? → best_state 저장
                  └─ 5 epoch 정체? → break (early stop)
   ↓
[Phase 3: 마무리]
   best 모델로 confusion matrix 생성
   metadata.json + learning_curves.png + confusion_matrix.png 저장
   ↓
   zip artifact:
       trained_models.zip ← out/ 폴더 통째로 압축
   ↓
   signature.txt 생성 (zip들의 SHA1)
```

#### D.2 `_build_loaders` 단계별

가장 디테일하게 봐야 하는 함수. 핵심 트릭들이 다 여기 있어요.

```python
def _build_loaders(directory, split, batch, seed):
    
    # ① 두 가지 transform 정의
    train_tf = Compose([
        Resize((256, 256), antialias=True),
        RandomHorizontalFlip(p=0.5),        # ← 매번 50% 확률로 뒤집기
        RandomRotation(degrees=15),          # ← 매번 -15~+15° 랜덤
        Normalize(mean=ImageNetMean, std=ImageNetStd),
    ])
    val_tf = Compose([
        Resize((256, 256), antialias=True),
        Normalize(mean=ImageNetMean, std=ImageNetStd),
        # ← augmentation 없음!
    ])
```

**핵심 트릭 1**: train과 val에 다른 transform을 적용해야 함. 그래서 LeafDataset을 2번 만듦.

```python
    # ② LeafDataset 2번 — 같은 폴더, 다른 transform
    train_full = LeafDataset(directory, transform=train_tf)
    val_full = LeafDataset(directory, transform=val_tf)
    labels = [lab for _, lab in train_full.samples]
```

데이터 구조 시각화:

```
train_full.samples = [
    (Path("images/Apple_Black_rot/image (1).JPG"), 0),  # idx 0
    (Path("images/Apple_Black_rot/image (2).JPG"), 0),  # idx 1
    ...
    (Path("images/Apple_healthy/image (1).JPG"),  1),
    ...
    (Path("images/Grape_spot/image (1076).JPG"),  7),   # idx 7227 (마지막)
]
val_full.samples = [정확히 같은 7,228개 (path, label)]  ← discover_classes가 sorted라 동일 순서
```

```python
    # ③ Stratified split — 같은 인덱스가 두 dataset에 통용
    train_idx, val_idx = train_test_split(
        list(range(len(labels))),            # [0, 1, 2, ..., 7227]
        test_size=1-split,                    # 0.2
        stratify=labels,                      # 클래스 비율 유지
        random_state=seed,                    # 42
    )
    # train_idx: [3, 12, 47, ...] (5,783개)
    # val_idx:   [0, 5, 23, ...]  (1,445개)
    
    train_ds = Subset(train_full, train_idx)  # train_tf 적용됨
    val_ds = Subset(val_full, val_idx)        # val_tf 적용됨
```

`Subset`은 원본 dataset에 인덱스 list만 씌운 view예요. 데이터 복사 없이 가벼움.

**핵심 트릭 2**: train_full과 val_full의 sample 순서가 동일하므로, train_test_split이 만든 인덱스(예: idx 47)가 양쪽 dataset에서 같은 사진을 가리킴. 단지 적용되는 transform만 달라요.

```python
    # ④ WeightedRandomSampler — 클래스 균형
    train_labels = np.array([labels[i] for i in train_idx])
    class_count = np.bincount(train_labels)
    # array([ 497, 1312,  220,  504,  944, 1106,  339,  861])
    
    sample_weights = 1.0 / class_count[train_labels]
    # 길이 5,783, 각 sample의 weight = 1/(그 클래스 크기)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),   # 5,783 — epoch당 뽑기 횟수
        replacement=True,                   # 중복 허용
    )
```

**계산 시각화**:

```
train_idx[0]에 있는 sample이 Apple_rust(class_count[2]=220)면
    sample_weights[0] = 1/220 = 0.00455

train_idx[1]에 있는 sample이 Apple_healthy(class_count[1]=1312)면
    sample_weights[1] = 1/1312 = 0.000762

→ Apple_rust 하나가 Apple_healthy 하나보다 6배 자주 뽑힘
→ 클래스 전체 weight 합 = 1로 같음 (8개 클래스 × 1 = 8.0)
→ 결과적으로 batch에 8 클래스가 거의 균등하게 등장
```

```python
    # ⑤ DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch,        # 32
        sampler=sampler,                    # 위에서 만든 WeightedRandomSampler
        num_workers=8,                      # 백그라운드 워커 8개로 다음 batch 미리 준비
        persistent_workers=True,            # epoch마다 워커 재생성 안 함
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch,
        shuffle=False,                      # val은 섞을 필요 없음
        num_workers=8, persistent_workers=True,
    )
```

`num_workers=8`이 중요. macOS는 fork가 비싸서 worker를 매 epoch 새로 만들면 큰 오버헤드. `persistent_workers=True`로 epoch 끝나도 워커 살려둠.

#### D.3 `LeafDataset.__getitem__`이 호출되는 순간 (online augmentation의 비밀)

DataLoader가 batch를 만들 때 내부적으로 이렇게 동작해요:

```
DataLoader(batch_size=32)
   ↓
sampler가 32개 인덱스를 weight에 따라 뽑음: [47, 1893, 47, 502, 1, 47, ...]
   ↓                                              ↑↑↑↑ 같은 인덱스 중복 OK!
worker 8개가 병렬로 각 인덱스에 대해:
   dataset[47] 호출
       ↓
       train_ds.__getitem__(47)
           ↓
           실제로는 Subset이라 train_full.__getitem__(train_idx[47]) 호출
               ↓
               # LeafDataset.__getitem__:
               path, label = self.samples[idx]
               img = Image.open(path).convert("RGB")          # 디스크에서 로드
               tensor = F.to_image(img)                        # PIL → tensor (3, H, W) uint8
               tensor = F.to_dtype(tensor, float32, scale=True) # uint8 → float32 [0,1]
               
               if self.transform is not None:
                   tensor = self.transform(tensor)             # ← train_tf 적용
                                                                 # 매번 새 random!
               return tensor, label
   ↓
8 worker가 가져온 32개 (tensor, label) tuple을 batch로 묶음
   ↓
collate: tensor 32개를 첫 차원으로 쌓아서 (32, 3, 256, 256) 만듦
         label 32개를 (32,) 벡터로
   ↓
DataLoader가 (x, y) 튀어내옴 — 학습 루프가 받음
```

**왜 매 epoch마다 같은 사진이 다르게 보이는가**:

```
sampler가 인덱스 47을 epoch 1에서도 뽑고 epoch 2에서도 뽑았다고 치자.

epoch 1 회차:
    __getitem__(47) → img 로드 → train_tf(img)
                                       ↓
                                  RandomHorizontalFlip.forward() 호출
                                       ↓
                                  random.random() < 0.5? → True (이번에 뒤집기)
                                       ↓
                                  RandomRotation.forward() 호출
                                       ↓
                                  random.uniform(-15, 15) → 8.3° (이번 회전)
                                       ↓
                                  변형된 tensor 반환

epoch 2 회차:
    __getitem__(47) → 같은 img 로드 → train_tf(img)
                                       ↓
                                  random.random() < 0.5? → False (이번엔 안 뒤집음)
                                       ↓
                                  random.uniform(-15, 15) → -3.1° (다른 각도)
                                       ↓
                                  완전히 다른 변형된 tensor 반환
```

핵심: `RandomHorizontalFlip`이랑 `RandomRotation`은 **stateless** — 호출할 때마다 새 random 값을 굴림. 그래서 같은 sample을 N번 호출하면 N개의 다른 변형이 나옴. 디스크에는 원본 1장만 있고요.

#### D.4 한 epoch이 어떻게 돌아가는지

```
def _epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None         # train epoch이냐?
    model.train(is_train)                     # BN/Dropout 모드 전환
    loss_sum, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(is_train):    # gradient 계산 on/off
        for x, y in loader:                   # DataLoader iterate
            x = x.to(device)                  # GPU/MPS로 이동
            y = y.to(device)
            
            logits = model(x)                 # ① forward
            loss = criterion(logits, y)       # ② loss
            
            if is_train:
                optimizer.zero_grad()         # ③ gradient 초기화
                loss.backward()               # ④ backward (역전파)
                optimizer.step()              # ⑤ weight 업데이트
            
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return loss_sum / total, correct / total
```

**한 epoch (181 step) 시간 흐름**:

```
Step 1:
  batch_1 = next(train_loader)  ← (32, 3, 256, 256) + (32,) 라벨
  x.to(device), y.to(device)    ← 메모리 이동 (CPU → MPS/CUDA)
  logits = model(x)             ← (32, 8) 8 클래스 logit
  loss = CrossEntropyLoss(logits, y)  ← 스칼라
  optimizer.zero_grad()
  loss.backward()               ← model.parameters()의 .grad 채워짐
  optimizer.step()              ← weight = weight - lr * .grad
  loss_sum += loss.item() * 32
  correct += 맞춘 갯수
  
Step 2:
  ... 반복
  
...

Step 181:
  ... 마지막 batch (5,783 mod 32 = 23장, 부족분은 sampler가 채움)
  
return: train_loss (총 loss / 5783), train_acc (총 correct / 5783)
```

train epoch 끝나면 같은 함수에 `optimizer=None` 넘겨서 val epoch:

```
Val epoch (46 step):
  with torch.set_grad_enabled(False):  ← gradient 계산 안 함
      for x, y in val_loader:
          logits = model(x)
          loss = criterion(logits, y)
          # 여기서 끝 — backward나 optimizer 호출 안 함
          loss_sum += ...
          correct += ...

return: val_loss, val_acc
```

#### D.5 forward / backward를 그림으로

ScratchCNN 모델 안에서 하나의 사진이 어떻게 변하는지:

```
입력 batch x: (32, 3, 256, 256)  ─ 32장 RGB
   │
   ▼
self.features (4 conv block)
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 1: _conv_block(3, 32)                              │
   │ │   Conv2d(3, 32, kernel=3, padding=1)  → (32, 32, 256, 256) │
   │ │   BatchNorm2d(32)                     → (32, 32, 256, 256) │
   │ │   ReLU                                → (32, 32, 256, 256) │
   │ │   Conv2d(32, 32, kernel=3, padding=1) → (32, 32, 256, 256) │
   │ │   BatchNorm2d(32)                     → (32, 32, 256, 256) │
   │ │   ReLU                                → (32, 32, 256, 256) │
   │ │   MaxPool2d(2)                        → (32, 32, 128, 128) │
   │ └──────────────────────────────────────────────────────────┘
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 2: _conv_block(32, 64)                             │
   │ │   ... 같은 패턴, 채널 32→64                                  │
   │ │   출력: (32, 64, 64, 64)                                   │
   │ └──────────────────────────────────────────────────────────┘
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 3: _conv_block(64, 128)                            │
   │ │   출력: (32, 128, 32, 32)                                  │
   │ └──────────────────────────────────────────────────────────┘
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 4: _conv_block(128, 256)                           │
   │ │   출력: (32, 256, 16, 16)                                  │
   │ └──────────────────────────────────────────────────────────┘
   │
   ▼
self.head
   │ AdaptiveAvgPool2d(1)  → (32, 256, 1, 1)    ─ 16×16 격자를 평균 1×1로
   │ Flatten()             → (32, 256)          ─ 사진당 256차원 벡터
   │ Dropout(0.4)          → (32, 256)          ─ 학습 중 40% 끔
   │ Linear(256, 8)        → (32, 8)            ─ 클래스 logit
   ▼
출력: logits (32, 8)
```

**Forward = 위에서 아래로 계산해 logits 만들기**.

**Backward = 거꾸로 위로 거슬러 올라가며 gradient 계산**:

```
loss (scalar) ← CrossEntropyLoss(logits, y)
   │
   │ ∂loss/∂logits 계산  → (32, 8)
   ▼
Linear(256, 8)        ← ∂loss/∂weight, ∂loss/∂bias 계산
   │ chain rule로 ∂loss/∂Linear_input 계산 → (32, 256)
   ▼
Dropout                ← 학습 중 어떤 뉴런을 껐는지 기억해서 그쪽만 통과
   ▼
Flatten                ← reshape 역연산 → (32, 256, 1, 1)
   ▼
AdaptiveAvgPool2d      ← 평균의 미분 → (32, 256, 16, 16)
   ▼
[Block 4 backward]     ← Conv, BN, ReLU 각각의 미분
   ▼
[Block 3 backward]
   ▼
[Block 2 backward]
   ▼
[Block 1 backward]
   ▼
입력에 대한 gradient (∂loss/∂x) — 학습엔 안 씀, 그냥 chain 끝점
```

PyTorch가 forward 중에 computation graph를 기억해뒀다가 `loss.backward()` 한 줄에 이 모든 단계를 자동 수행. 모든 `model.parameters()`의 `.grad` 속성이 채워짐.

`optimizer.step()`이 그 `.grad`를 보고:

```python
for p in optimizer.param_groups[0]['params']:
    p.data -= lr * p.grad   # Adam은 더 복잡한 식, 직관은 이거
```

각 weight를 gradient 반대 방향으로 살짝 옮김. **이걸 4,500번 반복하면 weight가 정답 쪽으로 충분히 옮겨가서 모델이 학습됨.**

#### D.6 학습 끝나면 어떻게 zip이 만들어지나

```
[학습 종료 후 train.py가 하는 일]

[1] best 모델로 confusion matrix 생성
       best_model.load_state_dict(artifacts[best_name]["state"])
       for x, y in val_loader:
           logits = best_model(x)
           y_true.extend(y.tolist())
           y_pred.extend(logits.argmax(1).tolist())
       cm = confusion_matrix(y_true, y_pred)
       confusion_matrix_plot(cm, classes, out / "confusion_matrix.png")
       (out / "classification_report.txt").write_text(
           classification_report(y_true, y_pred, target_names=classes)
       )

[2] learning_curves 시각화
       learning_curves(artifacts[best_name]["history"], out / "learning_curves.png")

[3] metadata.json 생성
       metadata = {
           "version": "1.0.0",
           "trained_at": "2026-05-21T...",
           "seed": 42,
           "classes": ["Apple_Black_rot", "Apple_healthy", ...],
           "class_to_idx": {"Apple_Black_rot": 0, ...},
           "image_size": 256,
           "normalize_mean": [0.485, 0.456, 0.406],
           "normalize_std":  [0.229, 0.224, 0.225],
           "models": {"scratch": {"val_accuracy": 0.9979, "best_epoch": 14}, ...},
           "split": {"train": 0.8, "val": 0.2},
       }
       (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

[4] artifacts/ 폴더 통째로 zip
       artifacts/
       ├── model_scratch.pt           (1.18M 파라미터, ~5MB)
       ├── model_transfer.pt          (4M 파라미터, ~16MB)  [opt-in]
       ├── metadata.json
       ├── learning_curves.png
       ├── confusion_matrix.png
       └── classification_report.txt
       ↓
       trained_models.zip (≈20MB)

[5] signature.txt 생성
       compute_sha1("trained_models.zip") → "101dd3f4..."
       compute_sha1("augmented_directory.zip") → "0506b961..."
       
       signature.txt 내용:
       101dd3f43b16d60e1e827558fb4e10b19ae396cc  trained_models.zip
       0506b961d81e1941fc9ca972988164467a560646  augmented_directory.zip
```

평가일에 USB로 trained_models.zip + augmented_directory.zip을 옮기면, `signature.txt`의 해시랑 비교해서 무결성 확인.

---

### E. predict.py 흐름

#### E.1 파일 지도

```
predict.py
    │  ① len(expanded) == 1 → 단일 모드 (figure 표시)
    │  ② len(expanded) > 1  → 다중 모드 (콘솔 표)
    │
    └─→ predictor.py::
            load_artifact   (zip 풀고 모델 로드 — 1번만)
            predict_one     (사진 1장 추론)
            render          (matplotlib figure)
```

#### E.2 단일 vs 다중 자동 분기

```
사용자: ./predict.py "images/Apple_rust/image (1).JPG"
   ↓
[1] paths = [Path("images/Apple_rust/image (1).JPG")]
   ↓
[2] _expand(paths)
       각 path 확인:
         is_dir() → True면 rglob("*.JPG")로 펼치고
         is_dir() → False면 그대로 append
       결과: expanded = [Path("images/Apple_rust/image (1).JPG")]  ← 1개
   ↓
[3] len(expanded) == 1 → 단일 모드
   ↓
[4] artifact = load_artifact("trained_models.zip", prefer="scratch")
       (아래 E.3에서 상세)
   ↓
[5] result = predict_one(artifact, expanded[0])
   ↓
[6] render(result, save=None)
       → matplotlib figure 표시
```

vs.

```
사용자: ./predict.py /tmp/test_images/Unit_test1/
   ↓
[1] paths = [Path("/tmp/test_images/Unit_test1/")]
   ↓
[2] _expand:
       Unit_test1/이 디렉토리 → rglob("*.JPG") → 10개 펼침
   결과: expanded = [Apple_healthy1.JPG, Apple_BlackRot2.JPG, ...]  ← 10개
   ↓
[3] len(expanded) > 1 → 다중 모드
   ↓
[4] artifact = load_artifact(...)   ← 모델 로드 한 번만!
   ↓
[5] for path in expanded:
       result = predict_one(artifact, path)
       true_cls = _guess_class_from_name(path.name, artifact.classes)
       (콘솔에 "OK   Apple_healthy (99.8%)  ← Apple_healthy1.JPG" 출력)
       if save_dir: render(..., save=save_dir / f"{path.stem}_pred.png")
   ↓
[6] Self-check: 10/10 = 100.00% 표시
```

#### E.3 `load_artifact` — zip 풀기 + 모델 로드

```python
def load_artifact(zip_path, prefer="scratch"):
    # ① zip을 같은 폴더의 숨김 디렉토리에 풂
    extract_dir = zip_path.parent / f".{zip_path.stem}_unpacked"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    
    # extract_dir/
    #   model_scratch.pt
    #   model_transfer.pt
    #   metadata.json
    #   learning_curves.png
    #   confusion_matrix.png
    #   classification_report.txt
    
    # ② metadata 읽기 (클래스 라벨!)
    metadata = json.loads((extract_dir / "metadata.json").read_text())
    classes = metadata["classes"]  # ["Apple_Black_rot", "Apple_healthy", ...]
    
    # ③ weight 파일 찾기 (fallback 로직)
    weight_file = extract_dir / f"model_{prefer}.pt"
    if not weight_file.exists():
        for alt in ("scratch", "transfer"):
            cand = extract_dir / f"model_{alt}.pt"
            if cand.exists():
                weight_file = cand
                prefer = alt
                break
    
    # ④ 빈 모델 만들고 weight 채움
    model = _build_model(prefer, num_classes=len(classes))
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()  # ← BN/Dropout을 평가 모드로 전환 (중요!)
    
    return LoadedArtifact(model=model, classes=classes, image_size=256, model_used=prefer)
```

`model.eval()` 호출이 중요. BN과 Dropout은 학습 중과 평가 중 동작이 달라요. eval 모드 안 하면 결과가 비결정적이 됨.

#### E.4 `predict_one` — 한 장 추론

```python
def predict_one(artifact, image_path):
    # ① 이미지 로드
    rgb = np.array(Image.open(image_path).convert("RGB"))  # (H, W, 3) uint8
    
    # ② 전처리 — train.py의 val_tf랑 정확히 같아야 함!
    tensor = _preprocess(rgb, size=artifact.image_size)
    # _preprocess 내부:
    #   img = Image.fromarray(rgb).resize((256, 256), BILINEAR)
    #   t = F.to_image(img)                              # (3, 256, 256) uint8
    #   t = F.to_dtype(t, float32, scale=True)           # [0, 1]
    #   t = F.normalize(t, mean=ImageNetMean, std=ImageNetStd)
    #   return t.unsqueeze(0)                             # (1, 3, 256, 256) — batch 차원 추가
    
    # ③ 추론 (gradient 계산 안 함)
    with torch.no_grad():
        logits = artifact.model(tensor)        # (1, 8)
        probs = torch.softmax(logits, dim=1)[0]  # (8,) 합쳐서 1.0
        idx = int(probs.argmax())              # 가장 큰 인덱스
    
    return {
        "class": artifact.classes[idx],       # "Apple_rust" 등
        "confidence": float(probs[idx]),       # 0.998 등
        "rgb": rgb,                            # figure에 보여줄 원본
        "transformed": mask_transform(rgb),    # figure에 보여줄 mask
        "model_used": artifact.model_used,    # "scratch" or "transfer"
    }
```

**train.py와 일관성이 중요한 이유**: 학습 분포와 추론 분포가 다르면 모델이 같은 의미의 입력을 못 받음. resize 방법(BILINEAR), normalize 통계(ImageNet mean/std)가 정확히 일치해야 정확도가 보장됨.

#### E.5 한 사진의 데이터 변환을 끝까지 추적

```
디스크: "Apple_rust1.JPG"
   ↓ Image.open + convert("RGB")
PIL Image (H_orig, W_orig, RGB)        ← 원본 크기 (가변)
   ↓ np.array
numpy (H_orig, W_orig, 3) uint8        ← [0, 255]
   ↓ Image.fromarray + resize((256, 256))
PIL Image (256, 256, RGB)               ← 표준 크기
   ↓ F.to_image
torch.Tensor (3, 256, 256) uint8        ← 채널 first
   ↓ F.to_dtype(float32, scale=True)
torch.Tensor (3, 256, 256) float32       ← [0, 1]
   ↓ F.normalize(mean, std)
torch.Tensor (3, 256, 256) float32       ← 평균 0, 분산 1 근처
   ↓ unsqueeze(0)
torch.Tensor (1, 3, 256, 256)            ← batch 차원 추가
   ↓ model(tensor)
torch.Tensor (1, 8)                      ← 8 클래스 logit
   ↓ softmax(dim=1)
torch.Tensor (1, 8)                      ← 합쳐서 1.0 확률
   ↓ [0]
torch.Tensor (8,)                        ← batch 차원 제거
   ↓ argmax()
int (예: 2)                              ← 가장 큰 인덱스
   ↓ classes[2]
str "Apple_rust"                          ← 클래스명
```

**전체 흐름이 머리에 들어와 있으면 코드 보면서 헷갈리는 일이 없어요.**

---

### F. 마무리 — 코드 한 줄 한 줄이 어디로 연결되나

가장 헷갈리는 부분 정리:

1. **`LeafDataset`이 어떻게 augmentation을 매번 새로 적용하나**:
   - `__getitem__`이 호출될 때 `self.transform(tensor)`가 실행됨.
   - `RandomHorizontalFlip`, `RandomRotation`은 stateless라서 호출마다 새 random.
   - DataLoader가 batch를 만들 때 __getitem__을 호출하니까 매 step, 매 epoch마다 다른 변형.

2. **`WeightedRandomSampler`가 실제로 무슨 일을 하나**:
   - DataLoader가 batch_size=32만큼 인덱스를 뽑을 때 `sampler.__iter__()`를 호출.
   - sampler는 weight에 비례한 확률로 인덱스를 뽑음 (replacement=True니까 중복 OK).
   - 결과적으로 batch 안에 작은 클래스(Apple_rust)가 자주, 큰 클래스(Apple_healthy)가 비례적으로.

3. **train.py의 `_build_loaders`에서 LeafDataset을 왜 2번 만드나**:
   - train_tf와 val_tf가 달라서. 한 dataset에 두 transform을 적용할 수 없음.
   - `discover_classes`가 sorted라 두 dataset의 sample 순서가 동일.
   - 그래서 train_test_split이 만든 인덱스 list가 양쪽에 통용됨.

4. **`metadata.json`이 왜 zip 안에 들어가나**:
   - 모델 weight만 있으면 클래스 라벨(0=Apple_Black_rot, 1=Apple_healthy, ...)을 모름.
   - metadata.json이 라벨 매핑을 보존해서 predict.py가 클래스명을 출력 가능.

5. **`signature.txt`가 왜 zip이 아닌 평문인가**:
   - 평가일에 `make verify`가 `shasum`이랑 `diff`로 즉시 비교할 수 있어야 함.
   - 표준 `<sha1>  <basename>` 형식이라 `shasum -c signature.txt`도 동작.

---

## 🔗 관련 문서
- 디자인 결정 + 근거: [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)
- 구현 plan: [docs/superpowers/plans/2026-04-28-leaffliction.md](superpowers/plans/2026-04-28-leaffliction.md)
