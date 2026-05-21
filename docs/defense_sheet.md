# Leaffliction — Defense Script (대본)

> 평가표(Intra Projects Leaffliction Edit) 항목 순서대로 시연하면서 입으로 읽을 수 있게 작성한 대본.
>
> 각 섹션 포맷:
> - **🎯 평가자가 확인하는 것** — 평가표 항목 인용
> - **📖 개념 — 이걸 왜 하나** — 단계의 의미·ML 워크플로우상 위치
> - **💻 명령** — 그대로 복붙
> - **🎤 대사** — 인용 박스를 입으로 읽으면 됨
> - **📂 관련 코드** — 어느 파일이 무엇을 하는지
> - **⚠️ 함정** — 0점 위험 + 자주 받는 후속 질문

---

## 0. 프로젝트 소개 (Intro)

### 📖 무엇을 만드는 프로젝트인가

**Leaffliction**은 잎 사진을 보고 어떤 식물의 어떤 질병(또는 건강)인지 자동 분류하는 컴퓨터 비전 프로젝트입니다. 42 Paris의 AI 입문 과제로, 머신러닝 전체 파이프라인(데이터 분석 → 전처리 → 모델 학습 → 추론)을 한 프로젝트에서 다 다룹니다.

**분류 대상 8 클래스**:

| 식물 | 상태 (4가지씩) |
|------|--------------|
| 사과 (Apple) | healthy / Black_rot / rust / scab |
| 포도 (Grape) | healthy / Black_rot / Esca / spot |

**데이터셋**: PlantVillage. 회색 통제 배경에서 촬영된 잎 단일 사진 7,221장. `images/`에 클래스명 폴더로 정리됨.

### 📖 왜 이 문제가 의미 있나

- **실제 농업 응용**: 농부가 휴대폰으로 잎 사진을 찍어 즉시 진단 → 농약 최적화, 조기 대응. 인도·아프리카에서 실제 모바일 앱으로 배포된 사례 있음.
- **컴퓨터 비전의 정석 분류 문제**: 클래스 경계 명확, small-scale, EDA부터 모델 평가까지 한 사이클을 끝낼 수 있는 규모.
- **ML 흔한 함정의 교과서**: 클래스 불균형(가장 큰 1640장 vs 가장 작은 275장 = 6배), 데이터 누수(augment-then-split의 함정), overfitting — 모두 이 데이터셋에서 자연스럽게 마주치는 문제들.

### 📖 4 Part 구성 — ML 워크플로우 매핑

| Part | 명령 | 표준 ML 단계 | 목적 |
|------|------|-------------|------|
| 1 | `Distribution.py` | **EDA** (Exploratory Data Analysis) | 데이터를 보기 — 클래스 균형 파악 |
| 2 | `Augmentation.py` | **Data Augmentation** | 클래스 불균형 보정 + 변형 다양성 확보 |
| 3 | `Transformation.py` | **Feature Visualization** | 잎의 어떤 특성이 분류에 쓰이는지 시각화 |
| 4 | `train.py` + `predict.py` | **Model Training + Inference** | CNN 학습 + 새 사진 예측 |

→ 산업의 실제 ML 프로젝트도 이 순서를 따릅니다. 우리는 PDF 시연 편의상 명령을 4개로 분리했지만 본질은 표준 supervised learning pipeline.

### 📖 우리의 핵심 설계 결정 (한 문장씩)

1. **모델은 ScratchCNN을 기본**, EfficientNet-B0 transfer는 비교용 opt-in — 모든 레이어를 직접 설계해 설명 가능성을 우선.
2. **학습은 원본 `images/`로 진행**, augmentation은 train batch에 메모리에서만 적용 — 데이터 누수 회피.
3. **클래스 불균형은 `WeightedRandomSampler`로 batch 단위에서 보정** — augmented_directory를 학습에 안 쓰고도 같은 효과.
4. **재현성을 위해 모든 random에 `seed=42`** — `eval_val.py`로 학습에 안 쓴 val 1,445장을 동일하게 재생 가능.

---

## 1. Error Management

### 🎯 평가자가 확인하는 것

> "Check that the signature contained in `signature.txt` is identical to that of the `.zip` file of the data set. **If signatures differ, the evaluation stops here.** Python인 경우 `flake8` norm 검사."

### 📖 개념 — 이걸 왜 하나

**signature.txt**: 학습이 완료된 모델 zip을 USB로 평가 컴퓨터에 옮기는 과정에서 누군가 (또는 사고로) 파일이 바뀌었는지 검증하는 단계. SHA1은 cryptographic hash 함수라 입력 1 byte만 달라도 완전히 다른 출력을 냄. PDF Chapter V가 "0점 사유"로 명시한 항목.

**flake8 norm**: 42 학교는 코드 스타일을 평가표에서 강제. PEP8(Python style guide) 위반이 1줄이라도 있으면 **Norm flag → 즉시 0점**. ruff(format)와 flake8(lint) 둘 다 통과해야 안전.

### 💻 명령

```bash
make verify                          # signature.txt와 zip의 SHA1을 diff
flake8 src tests *.py                # PDF가 명시한 norm 도구
```

수동으로 보고 싶으면:

```bash
shasum trained_models.zip augmented_directory.zip
cat signature.txt
```

### 🎤 대사

`make verify`가 깨끗이 끝나면:

> "Signature 검증부터 하겠습니다. `signature.txt`에는 학습 직후 두 zip의 SHA1 해시가 기록돼 있고, 지금 다시 계산한 해시와 `diff`해서 차이가 없으면 zip이 학습 시점 그대로라는 증거입니다. PDF Chapter V가 '0점 사유'로 명시한 항목이고, 우리는 통과입니다."

`flake8`이 출력 없이 끝나면:

> "Norm 검사도 통과입니다. ruff(format)와 flake8(lint) 둘 다 pre-commit hook에 걸려 있어서 commit 시점에 자동 검사됩니다."

### 📂 관련 코드

- **`src/leaffliction/signature.py`** — `sign()`은 SHA1 계산 후 `signature.txt` 작성, `verify()`는 zip을 다시 해싱해 비교. 표준 `hashlib`만 씀.
- **`train.py` 마지막** — 학습이 끝나면 두 zip(`trained_models.zip`, `augmented_directory.zip`)의 해시를 자동 기록.
- **`scripts/verify.sh`** — `shasum *.zip`과 `signature.txt`의 diff. `make verify`가 호출.
- **`Makefile`** — `lint`, `verify`, `test`, `format`, `smoke` 단축 명령 모음.

### ⚠️ 함정

- Signature 불일치 → **평가 즉시 종료, 0점**. 백업 USB로 재시도.
- flake8 에러 1개 → **Norm flag, 0점**. `make lint`로 사전 확인 필수.

---

## 2. Part 1 — Analysis of the Data Set

### 🎯 평가자가 확인하는 것

> "Read the code, Run the code and pie chart as in the subject must appear: `./Distribution.[extension] ./Apple`"

### 📖 개념 — 이걸 왜 하나

**EDA(Exploratory Data Analysis)**는 모든 ML 프로젝트의 첫 단계입니다. 모델 설계, augmentation 전략, 평가 방식 모두 데이터의 분포에 따라 달라지기 때문에 **데이터를 보지 않고 모델부터 만드는 건 ML 안티패턴**입니다.

우리가 봐야 할 것:
- **클래스별 사진 수** — 불균형이 있으면 모델은 다수 클래스로 편향. WeightedSampler/oversampling/loss weighting 등 대응 필요.
- **클래스 수** — 8개. softmax + cross-entropy의 표준 구조.
- **데이터 양** — 7,221장. small-scale이지만 augmentation으로 보강 가능.

**pie chart vs bar chart의 차이**:
- pie는 **비율** 한눈에 — Apple_healthy가 전체의 22%
- bar는 **절대 개수** 정확히 — Apple_rust 275 vs Apple_healthy 1640
- 두 개 다 보여줘야 직관적

### 💻 명령

```bash
./Distribution.py ./images
```

옵션 확인:

```bash
./Distribution.py --help
./Distribution.py ./images --save /tmp/dist.png   # 차트 저장
```

### 🎤 대사

명령 실행 직전:

> "Part 1은 EDA, 데이터셋 분석입니다. `Distribution.py`가 `images/` 안의 모든 하위 폴더를 하나의 클래스로 보고 사진 수를 세서 pie chart와 bar chart 두 개로 시각화합니다. ML 프로젝트의 시작점이고, 데이터 분포를 봐야 다음 단계 설계가 가능합니다."

차트가 뜨면 (8 클래스 동시에):

> "8개 클래스 총 7,221장입니다. pie chart로 비율을, bar chart로 정확한 개수를 같이 봅니다. `Apple_healthy`가 1640장으로 가장 많고 `Apple_rust`가 275장으로 가장 적습니다. **약 6배 불균형**이고, 이게 그대로 Part 2 data augmentation의 동기가 됩니다."

> "막대 위에 숫자 라벨을 추가해서 정확한 카운트를 즉시 볼 수 있게 했습니다."

### 📂 관련 코드

- **`src/leaffliction/dataset.py`**
  - `discover_classes(root)` — `pathlib.Path.rglob("*.JPG")`로 모든 사진을 찾고 부모 폴더 이름(=클래스)으로 그룹핑. 결과는 `{class_name: [Path, Path, ...]}` dict.
  - 같은 함수가 Part 2 augmentation, Part 4 train.py에서도 클래스 라벨 매핑에 재사용. **단일 진실의 출처(single source of truth)** 패턴.
- **`src/leaffliction/viz.py`**
  - `plot_distribution(counts)` — matplotlib + seaborn으로 pie + bar를 한 figure에 grid 배치. bar 위에 `ax.annotate`로 숫자 표시.
- **`Distribution.py`** — typer 기반 CLI wrapper. 위 두 함수를 조합해 화면에 표시하거나 `--save`로 PNG 저장.

### ⚠️ 함정

- pie chart 모양이 subject PDF 예시와 시각적으로 동일해야 함 → 우리는 일치.
- 평가자가 "왜 두 가지 차트?"라고 물으면: pie는 비율, bar는 절대수. EDA의 표준 dual visualization.

---

## 3. Part 2 — Data Augmentation

### 🎯 평가자가 확인하는 것

> "6 images as in the subject must appear. `./Augmentation.[ext] ./Apple/apple_healthy/image (1).JPG` and 6 versions of the same image must have been created: `image (1)_Flip.JPG`, `_Rotate.JPG`, `_Skew.JPG`, `_Shear.JPG`, `_Crop.JPG`, `_Distortion.JPG`."

### 📖 개념 — 이걸 왜 하나

**Data Augmentation**은 컴퓨터 비전의 핵심 기법입니다. 원리:

1. **물리적 invariance(불변성)**: 사과 잎의 질병은 좌우 뒤집어도 같은 질병. 회전해도 같은 질병. 즉 **모델이 학습해야 할 라벨은 이런 변형에 불변**. 따라서 변형된 사진도 같은 라벨로 학습 데이터에 추가하면 효과적.
2. **데이터 부족 보강**: PlantVillage 같은 small-scale은 학습 데이터가 부족 → augmentation이 사실상 무한 데이터 효과.
3. **Overfitting 완화**: 모델이 특정 사진의 픽셀 위치를 외우는 걸 막음.

**6 op이 왜 이 6개인가**:
- **기하 변형(Geometric)**: Flip, Rotate, Skew, Shear, Crop — 잎의 위치·방향·크기 다양성
- **광학 변형(Optical)**: Distortion — 카메라 렌즈 왜곡 흉내

→ 실제 농부가 휴대폰으로 찍을 때 발생할 수 있는 **현실적 변형 분포**를 흉내.

### 💻 명령

```bash
# 단일 모드 — PDF 시연용
./Augmentation.py "images/Apple_healthy/image (1).JPG"
ls "images/Apple_healthy/image (1)"*

# 배치 모드 (Part 1 verify 단계에서 검증)
./Augmentation.py images/
```

### 🎤 대사

명령 실행 직전:

> "Part 2는 data augmentation입니다. PDF가 명시한 6개 변형 — Flip, Rotate, Skew, Shear, Crop, Distortion — 을 한 사진에 각각 적용해 6개의 sibling 파일을 만듭니다."

`ls` 결과 보여주며:

> "원본 옆에 `image (1)_Flip.JPG`처럼 suffix만 붙여 6개가 생성됐습니다. PDF 파일명 규칙과 정확히 일치합니다."

각 변환 의미 (코드 보여주며):

> "구현은 Albumentations 라이브러리입니다."
> - **Flip** = `HorizontalFlip` — 좌우 반전 (잎의 좌우 대칭성)
> - **Rotate** = `Rotate(limit=30°)` — ±30도 회전 (촬영 각도 다양성)
> - **Skew** = `Affine(shear x+y, ±15°)` — x·y 양방향 평행사변형 변형 (원근 왜곡)
> - **Shear** = `Affine(shear x, ±25°)` — x축만 더 강한 shear (다른 강도 비교용)
> - **Crop** = `RandomResizedCrop(70~100%)` — 부분 잘라 256×256으로 확대 (잎이 화면을 채우는 비율 다양성)
> - **Distortion** = `OpticalDistortion(±0.4)` — 렌즈 배럴/핀쿠션 왜곡 (광학 변형)

> "각 변형은 PIL/Numpy array를 입력으로 받고 같은 shape의 변형된 array를 출력합니다."

시연 끝나면:

```bash
rm "images/Apple_healthy/image (1)_"*.JPG
```

> "다음 단계 영향 없게 6 sibling은 정리하겠습니다."

### 📂 관련 코드

- **`src/leaffliction/augment.py`**
  - `AUGMENTATION_OPS` — 6 op을 dict로 정의. key가 suffix 이름(`Flip`), value가 Albumentations transform 객체.
  - `apply_op(name, image)` — 단일 op 실행. 단일 모드(`Augmentation.py` 단일 파일)와 배치 모드 둘 다 호출.
  - `balance_directory(src, dst, target_count)` — 배치 모드의 핵심. (1) 원본 복사 (2) 부족분만큼 6 op 중 random 적용해 채움 (3) 끝에 `zip_directory()`로 `augmented_directory.zip` 생성.
  - `zip_directory(directory)` — Python 표준 `zipfile.ZIP_DEFLATED`로 압축. PDF Chapter V가 signature 비교용 zip을 요구하기 때문에 자동 생성.
- **`Augmentation.py`** — typer wrapper. 인자가 파일이면 단일 모드, 디렉토리면 배치 모드 자동 dispatch.

### ⚠️ 함정

- 파일명이 평가표 예시(`_Flip`, `_Rotate` 등)와 정확히 일치해야 함 — 우리는 일치.
- "원본 안 지웠지?" 묻는 평가자 있음 → 원본은 그대로, 6 sibling만 추가.

---

## 4. Part 1 추가 검증 — augmented_directory 균형 ⚠️

### 🎯 평가자가 확인하는 것

> "Now use Part 1 program on `augmented_directory`. **Each part of the pie chart must be equal.** If the pie chart is still the same as above you must put 0 to this exercise and exercise Part 1."

→ **불균형하면 Part 1, Part 2 둘 다 0점**. 평가표가 명시적으로 경고한 함정.

### 📖 개념 — 이걸 왜 하나

원본 `images/`는 클래스 불균형(6배). 그대로 학습하면:
- 모델이 다수 클래스(Apple_healthy 1640)로 편향 → 그냥 "Apple_healthy"만 외워도 22% 정확도 확보
- 소수 클래스(Apple_rust 275)는 학습 신호가 약해서 거의 무시

**해결 1 — Oversampling으로 디스크에 균형 데이터셋 생성**:
- 각 클래스를 가장 큰 클래스 크기에 맞춤 → 8 × 1640 = 13,120장
- 이게 `augmented_directory`. **이 데이터셋 자체는 학습에 쓰지 않음** (누수 이슈), Part 2의 PDF 요구사항 + zip + signature를 위해 만들 뿐.

**해결 2 — WeightedRandomSampler로 batch 단위 균형** (우리 학습에서 실제 사용):
- 디스크 데이터셋은 원본 그대로
- batch 만들 때 작은 클래스를 자주 뽑음
- 같은 효과 + 누수 회피

→ 평가표는 디스크 균형(해결 1)을 보지만, 실제 학습은 sampler(해결 2)를 씀. 두 가지가 양립.

### 💻 명령

```bash
# augmented_directory.zip이 아직 안 풀려있다면
unzip -q augmented_directory.zip

./Distribution.py ./augmented_directory
```

### 🎤 대사

명령 실행 직전:

> "이번엔 같은 `Distribution.py`를 `augmented_directory`에 실행합니다. 평가표가 명시적으로 경고한 단계 — augmented_directory가 원본과 동일하게 불균형하면 Part 1, Part 2 둘 다 0점입니다."

차트가 뜨면:

> "8 조각이 정확히 균등합니다. 8 × 1640 = 13,120장. 가장 많았던 `Apple_healthy`(1640)에 맞춰 나머지 7 클래스를 augmentation으로 1640까지 채웠습니다. pie chart 비율이 8 × 12.5% = 100%."

### 📂 관련 코드

- **`src/leaffliction/augment.py::balance_directory`** — 핵심 함수. 알고리즘:
  1. `discover_classes(src_root)`로 클래스별 사진 list 수집
  2. `target = max(len(p) for p in classes.values())`로 가장 큰 클래스 크기 계산
  3. 각 클래스에 대해:
     - 원본을 `dst_root/<class>/`로 복사
     - 부족분만큼 random op 적용한 augmented 파일 생성
  4. `zip_directory(dst_root)`로 압축
- **`Augmentation.py`** 배치 모드 — 디렉토리 인자 받으면 `balance_directory` 호출.

### ⚠️ 함정

- **0점 함정 1순위**. 이 단계에서 pie가 원본과 같아 보이면 Part 1 + Part 2 동시 무효.
- "왜 augmented_directory를 학습에 안 썼냐?" → §7 데이터 누수 설명 한 번 더.

---

## 5. Part 3 — Image Transformation

### 🎯 평가자가 확인하는 것

> "Read the code, Run the code and 6 images as in the subject must appear. The techniques used must be able to extract the characteristics of the plants, you can ask for explanations on each one of the transformations."

### 📖 개념 — 이걸 왜 하나

**Feature visualization**(특징 시각화)은 모델이 "잎의 어떤 면을 보고 분류하는지" 인간이 이해하기 위한 단계. plantCV(Plant Computer Vision)는 식물학 도메인에 특화된 영상처리 라이브러리.

**6 변환의 의미**:

1. **Original** — 입력 RGB. 베이스라인.
2. **Gaussian blur** — 작은 노이즈 완화 → mask가 더 깔끔해지는 전처리.
3. **Mask** — 배경 분리. 잎과 회색 배경을 분할해서 모델/후속 처리가 잎만 보게 함. **분류 정확도와 직결**.
4. **ROI(Region of Interest)** — mask + 경계 박스. 잎의 위치와 영역 표시.
5. **Analyze object** — 잎 모양 분석(면적, 둘레, 중심, 너비/높이 비). 식물학적 metric.
6. **Pseudolandmarks** — 잎 가장자리 따라 자동 keypoint. 모양 비교의 표준 도구.

**왜 9채널 color histogram이 같이 나오나**:
- 색 정보는 잎 분류의 큰 단서 — 갈색 병변(Black_rot), 노란 점(rust)
- RGB만 보면 부족 → HSV(색조/채도/명도)와 LAB(인간 시각 균등) 같이 봐서 9 채널
- 클래스별로 histogram 분포가 다름 = 색만으로도 일부 분류 가능

### 💻 명령

```bash
# 단일 모드 — 한 figure에 6 변환 + histogram
./Transformation.py "images/Apple_healthy/image (1).JPG"

# 옵션 확인 (PDF 명시 -h)
./Transformation.py -h

# 배치 모드 — 디렉토리 전체를 dst에 저장
./Transformation.py -src images/Apple_healthy -dst /tmp/transformed -mask
```

### 🎤 대사

명령 실행 직전:

> "Part 3은 plantCV로 잎의 특징을 시각화합니다. 한 figure 위쪽에 6 변환 패널, 아래쪽에 9채널 color histogram을 배치했습니다. plantCV는 식물 영상 처리에 특화된 도메인 라이브러리입니다."

화면 뜨면 위쪽 6 패널 하나씩 가리키며:

> 1. **Original** — 원본 RGB 그대로.
> 2. **Gaussian blur** — 3×3 가우시안 필터. 작은 노이즈를 평활화해서 다음 단계 mask가 정확해집니다.
> 3. **Mask** — 잎과 배경의 분리. 이게 가장 핵심입니다. **LAB 색공간에서 chroma magnitude**를 계산하고 Otsu threshold로 자동 cutoff를 잡습니다.
> 4. **ROI** — Region of Interest. mask 위에 경계 박스 overlay. 잎의 위치 시각화.
> 5. **Analyze object** — 잎 윤곽 분석. 면적·둘레·중심·종횡비를 출력.
> 6. **Pseudolandmarks** — 잎 가장자리를 따라 자동으로 keypoint를 잡음. 모양 비교에 쓰임.

mask 깊이 설명 (자주 묻는 질문):

> "초기엔 HSV saturation으로 threshold를 잡았는데 healthy 사과 잎(회색-녹색)에서 saturation이 너무 낮아 절반밖에 못 잡았습니다. **LAB 색공간의 chroma magnitude**로 바꿨습니다. LAB는 a축 = 녹↔적, b축 = 청↔황. 회색 배경은 a≈128, b≈128로 chroma `sqrt((a-128)² + (b-128)²)`가 0에 가깝고, 녹색 잎과 갈색 병변은 모두 chroma가 큽니다. 그 뒤 Otsu로 자동 threshold, morphological opening으로 노이즈 제거, 가장 큰 connected component만 유지, scipy `binary_fill_holes`로 잎 내부 작은 구멍을 메웁니다."

histogram 가리키며:

> "아래는 9채널 color histogram입니다. RGB 3 + HSV 3 + LAB 3 = 9. 클래스별로 분포가 다르기 때문에 색만 봐도 일부 분류가 가능합니다. 모델은 이런 색 정보 + 텍스처 + 모양을 모두 종합해서 판단합니다."

### 📂 관련 코드

- **`src/leaffliction/transform.py`**
  - `_binary_mask(rgb)` — mask 생성 핵심. LAB → chroma → Otsu → opening → largest CC → fill_holes 파이프라인.
  - `gaussian_blur`, `mask`, `roi`, `analyze_object`, `pseudolandmarks` — 6 변환 함수. 각자 plantCV 또는 OpenCV 호출.
  - `color_histogram(rgb)` — 9 채널 hist 계산.
- **`src/leaffliction/viz.py::plot_transformations`** — matplotlib `subplot2grid((2, 6), ...)`로 위 6 변환 + 아래 6칸 span hist 배치.
- **`Transformation.py`** — typer wrapper. 단일/배치 모드 자동 dispatch.

### ⚠️ 함정

- plantCV의 기본 `threshold.binary`는 단일 채널 입력이라 어떤 채널을 쓸지가 핵심 문제. 우리는 LAB chroma를 직접 만들어 넣음.
- "plantcv 함수가 아니라 직접 만들었어?" → "plantcv가 제공하는 building block(Otsu, fill_holes, fill 등)을 쓰되 mask의 robust한 입력 채널은 우리가 LAB chroma로 직접 정의했습니다."

---

## 6. Part 4 (1/4) — Classification 정확도 (≥90%)

### 🎯 평가자가 확인하는 것

> "Ask the student to run his program on a test set of minimum 100 images, the result of good prediction must be over 90%."

### 📖 개념 — 이걸 왜 하나

**모델 평가의 핵심 원칙**: 학습에 쓴 데이터로 평가하면 안 됨. 학생이 외운 답을 평가하는 거지 진짜 일반화 능력 측정이 아님. 그래서 데이터셋을 **train / val(또는 test) 두 부분으로 미리 나누고**, train으로 학습 → val로 평가가 표준.

**우리 구조**:
- `train.py`가 학습 시작 직전에 `sklearn.train_test_split(stratify=labels, random_state=42)`로 80:20 분할
- train 5,776장 학습 / val 1,445장은 **학습 중 한 번도 안 봄**
- `eval_val.py`가 같은 `seed=42`로 split을 재현 → 학습에 안 쓴 1,445장을 정확히 다시 뽑아 forward만 돌림

→ 같은 seed로 stratified split을 재현할 수 있다는 게 핵심. 학습과 평가가 같은 val set을 보장.

### 💻 명령

```bash
./scripts/eval_val.py ./images
./scripts/eval_val.py ./images --model transfer
```

### 🎤 대사

명령 실행 직전:

> "Part 4 첫 번째 항목은 100장 이상에서 90% 이상 정확도 요구입니다. `eval_val.py`는 train.py가 학습 시 `random_state=42`로 stratified 80/20 split을 했다는 사실을 이용해, 같은 seed로 split을 재현합니다. 즉 학습에 한 번도 안 본 1,445장(전체의 20%)을 정확히 다시 뽑아 모델 forward만 돌립니다. 100장보다 14배 많은 표본으로 측정합니다."

결과 출력되면:

> "1442/1445, **99.79%**입니다. PDF 요구 90%를 약 10pp 상회. 8 클래스 per-class breakdown도 같이 나오는데 모든 클래스가 99% 이상입니다."

transfer 결과:

> "비교용 transfer 모델은 99.86%로 약간 더 높습니다. ScratchCNN과 0.07pp 차이라 데이터셋이 명확해서 두 모델이 비슷하게 잘 푼 거지 한쪽의 트릭이 아닙니다."

### 📂 관련 코드

- **`scripts/eval_val.py`** — 평가 day 자동화 스크립트.
  - `LeafDataset(images, transform=val_tf)` — val 전용 transform(resize + normalize, augmentation 없음)으로 데이터셋 로드.
  - `train_test_split(... stratify=labels, random_state=42, test_size=0.2)` — train.py와 정확히 같은 분할 재현.
  - `Subset(full, val_idx)` — val 인덱스만 추출.
  - 모델 weight를 `trained_models.zip`에서 unzip해 load.
  - `DataLoader`로 forward → argmax → accuracy.
  - per-class breakdown까지 출력.

### 📌 참고 — val에서 실제로 틀린 사진 (defense day 사전 숙지용)

평가자가 "어떤 사진이 틀렸어요?"라고 물어볼 때 즉답 가능하도록.

**ScratchCNN — 3장 (1442/1445 = 99.79%)**:

| 정답 | 예측(틀림) | 신뢰도 | 파일 |
|------|-----------|--------|------|
| Grape_spot | Apple_scab | 57.1% | `images/Grape_spot/image (128).JPG` |
| Apple_healthy | Apple_Black_rot | 39.7% | `images/Apple_healthy/image (1040).JPG` |
| Apple_healthy | Grape_healthy | 44.1% | `images/Apple_healthy/image (1326).JPG` |

**TransferModel — 2장 (1443/1445 = 99.86%)**:

| 정답 | 예측(틀림) | 신뢰도 | 파일 |
|------|-----------|--------|------|
| Apple_healthy | Apple_scab | 99.1% | `images/Apple_healthy/image (1040).JPG` |
| Grape_Black_rot | Grape_Esca | 93.7% | `images/Grape_Black_rot/image (56).JPG` |

**관찰**:
- `Apple_healthy/image (1040).JPG`가 **두 모델 모두 틀림** → 데이터셋 자체의 어려운 사진(혹은 라벨링 노이즈) 가능성.
- ScratchCNN은 틀릴 때 **39~57% 낮은 confidence** = 망설였다는 정직한 신호.
- TransferModel은 틀릴 때도 **93~99% confidence** = 자신만만하게 틀림 (신경망의 전형적 overconfidence).
- → "두 모델 차이"를 단순 정확도가 아닌 **calibration 관점**에서 설명 가능.

### ⚠️ 함정

- "이거 train accuracy 아니냐?" → 아니. `seed=42` 재현으로 학습에 한 번도 안 본 1,445장.
- "100장만 보자" → 1,445장이 충분조건. 원하면 `--max 100`도 추가 가능.

---

## 7. Part 4 (2/4) — 모델 설명 (5점 단일 항목, 핵심 섹션)

### 🎯 평가자가 확인하는 것

> "The student must be able to explain the machine learning model he has chosen and how it works. **0 if the student can't explain. 5 if explanations are fluid.**"

→ **단일 항목 0~5점**. 평가표 전체에서 가장 큰 단일 점수 항목. 막히면 큰 손실.

### 📖 설명 순서 (7 sub-section)

1. **왜 CNN인가** — 문제 정의에서 시작
2. **데이터 흐름** — split → sampler → augmentation
3. **모델 구조** — ScratchCNN conv block / head
4. **학습 루프** — PyTorch 5줄 + autograd
5. **Loss / Optimizer / Scheduler / Early stopping**
6. **결과 시각화** — learning curve, confusion matrix
7. **데이터 누수 회피 + Transfer 비교** — 마무리

### 💻 보조 명령 (대사 중간에 코드/차트 띄우기)

```bash
cat src/leaffliction/models/scratch_cnn.py     # 모델 구조
cat src/leaffliction/trainer.py | head -100    # 학습 루프
open artifacts/learning_curves.png             # 학습 곡선
open artifacts/confusion_matrix.png            # 혼동 행렬
cat artifacts/classification_report.txt        # precision/recall/f1
cat artifacts/metadata.json                    # best_epoch + val_accuracy
```

---

### 7-1. 왜 CNN인가

> "이 문제는 8 클래스 leaf disease 분류이고 입력은 256×256 RGB 사진입니다. 사진 분류는 **CNN(Convolutional Neural Network)**이 표준입니다. CNN은 작은 필터로 이미지를 훑어 패턴을 찾고 여러 layer를 거치며 점점 추상적인 의미를 추출하는 신경망입니다."

> "직관: 사람이 그림 볼 때처럼 4단계로 추상화합니다. 처음엔 선·점 → 그 다음 모서리·텍스처 → 그 다음 잎맥·반점 → 마지막은 잎의 종류·질병. 우리 CNN의 4개 conv block이 이 4단계를 자동으로 학습합니다. **무엇이 분류에 중요한지 사람이 정해주지 않고 데이터에서 자동 발견**하는 게 deep learning의 본질입니다."

### 7-2. 데이터 흐름 — 세 단계

> "데이터는 학습 시작 전 세 단계의 전처리를 거칩니다."

**① Stratified 80/20 split**:
> "`sklearn.train_test_split(stratify=labels, random_state=42)`. 클래스 비율을 유지하면서 8:2로 나눕니다. Apple_healthy 1640장이면 train 1312 / val 328. 모든 클래스가 8:2를 지킵니다. `random_state=42`로 재현성 보장 — 같은 seed면 항상 같은 분할."

> "왜 stratified? 일반 random split이면 운나쁘게 val에 Apple_rust가 0장 들어갈 수도 → 측정 불가."

**② WeightedRandomSampler — train set 클래스 균형**:
> "Train set 안에서도 클래스 불균형이 남습니다(Apple_rust 220 vs Apple_healthy 1312). `WeightedRandomSampler`로 각 sample의 weight = `1 / 그 클래스 크기`를 줍니다. Apple_rust 한 sample = 1/220, Apple_healthy 한 sample = 1/1312. → batch 만들 때 작은 클래스가 자주 뽑힘. 결과적으로 batch마다 클래스가 거의 균등."

> "이게 augmented_directory를 학습에 안 쓰고도 클래스 균형을 맞추는 방법입니다."

**③ Online augmentation — train batch에만**:
> "Train batch에만 `RandomHorizontalFlip(p=0.5)`와 `RandomRotation(15°)`를 적용합니다. 메모리에서 batch 만들 때 매번 random하게 변형되니까 같은 이미지가 epoch마다 다르게 보이고, 디스크엔 저장 안 됩니다. **val batch엔 augmentation 없음** — 정직한 측정을 위해."

### 7-3. 모델 구조 — ScratchCNN

`src/leaffliction/models/scratch_cnn.py` 보여주며:

> "모델은 두 부분입니다. `self.features`(conv block 4개 — 특징 추출)와 `self.head`(GAP + Dropout + Linear — 분류)."

> "한 `_conv_block(in_ch, out_ch)`은 7층입니다: Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2)."

**Conv2d 설명**:
> "`Conv2d(in_ch=3, out_ch=32, kernel_size=3)`는 3×3 짜리 학습 가능한 필터 32개로 입력을 훑어 32개의 새 채널을 만듭니다."

> "**채널 개념**: 채널 1개 = 같은 사진을 한 가지 관점에서 본 흑백 지도. 처음엔 R, G, B 3개. Conv 통과하면 32 → 64 → 128 → 256으로 채널이 증가하고 각 채널이 '다른 종류의 패턴이 어디 있나' 지도가 됩니다. 필터 내용(스탬프)은 우리가 정하지 않고 학습으로 자동 결정."

**BatchNorm / ReLU / MaxPool**:
> "**BatchNorm2d**: 출력을 평균 0, 분산 1로 정규화 → 학습 안정. 입력 밝기·대비가 달라져도 일관되게."
>
> "**ReLU**: `f(x) = max(0, x)`. 음수→0, 양수→그대로. 신경망에 **비선형성** 부여. ReLU 없으면 layer 아무리 쌓아도 결국 선형 모델(직선 하나)."
>
> "**MaxPool2d(2)**: 2×2 영역에서 최대값만 남김. 공간 해상도가 절반(256→128→64→32→16). 작은 디테일 버리고 큰 그림에 집중."

**Block 진행**:
> "4 block을 거치면 (3, 256, 256)이 (256, 16, 16)으로 변환됩니다. 채널이 늘면서 공간이 줄어드는 깔때기 구조 — CNN의 전형입니다."

**Head**:
> "Head는 4층입니다."

```python
nn.AdaptiveAvgPool2d(1)  # GAP — (256, 16, 16) → (256, 1, 1)
nn.Flatten()             # → (256,)
nn.Dropout(0.4)          # 학습 중 40% 끔
nn.Linear(256, 8)        # 8 클래스 logit
```

> "**`AdaptiveAvgPool2d(1)`** = GAP(Global Average Pooling). 마지막 conv 출력 16×16 격자에서 위치 정보를 평균. '사진 전체에 256개 패턴이 얼마나 강한가'라는 256차원 벡터로 압축. 옛날 FC layer가 너무 무거웠던 문제를 해결하고 위치 invariance까지 얻습니다."

> "**`Dropout(0.4)`**: 학습 중 40% 뉴런 무작위로 끔. 특정 뉴런 의존을 막아 overfitting 완화."

> "**`Linear(256, 8)`**: 256 → 8 클래스 logit. softmax 적용하면 확률."

> "총 파라미터는 약 1.18M개."

### 7-4. 학습 루프 — PyTorch 5줄

`src/leaffliction/trainer.py` 보여주며:

> "학습 루프의 핵심은 PyTorch 표준 5줄입니다:"

```python
logits = model(x)              # ① forward — 예측
loss = criterion(logits, y)    # ② loss — 정답과 차이
optimizer.zero_grad()          # ③ 이전 gradient 지움
loss.backward()                # ④ backward — chain rule로 미분 자동 계산
optimizer.step()               # ⑤ weight 살짝 업데이트
```

> "forward에서 모델이 예측, criterion이 loss 계산, **`loss.backward()` 한 줄이 chain rule을 자동 적용**해 수백만 weight의 미분을 한 번에 계산. `optimizer.step()`이 그 반대 방향으로 weight를 살짝 옮김. 이걸 batch 단위로 25 epoch × 약 180 step = 약 4,500번 반복하면서 weight가 정답에 가까워집니다."

> "Autograd가 backward 자동화의 핵심. 우리는 forward만 정의하면 PyTorch가 chain rule을 자동으로. 손으로 작성하면 수천 줄 + 버그 천국인 작업입니다."

### 7-5. Loss / Optimizer / Scheduler / Early stopping

> "**`nn.CrossEntropyLoss`** — 분류의 표준 loss. 정답 클래스 확률이 1에 가까울수록 loss → 0, 멀수록 loss ↑. 내부적으로 logit → log_softmax → NLL을 한 번에."

> "**`optim.Adam(lr=1e-3, weight_decay=1e-4)`** — weight마다 적정 step 크기를 적응적으로 조절(momentum + RMSprop 결합). SGD보다 빠른 수렴. `weight_decay`는 L2 regularization으로 overfitting 완화."

> "**`ReduceLROnPlateau(factor=0.5, patience=2)`** — val_accuracy가 2 epoch 정체되면 learning rate를 절반으로. 학습 후반 미세조정에 효과."

> "**Early stopping (`patience=5`)** — val_accuracy가 5 epoch 동안 개선 없으면 학습 중단. Overfitting 진입 직전 차단."

> "→ 네 가지 모두 표준 ML 기법. overfitting 방지 + 학습 효율 향상."

### 7-6. 결과 시각화

`learning_curves.png` 보여주며:

> "학습 곡선입니다. train_loss와 val_loss가 함께 감소하며 수렴합니다. 어떤 epoch는 train_loss > val_loss인 경우도 있는데, train에 augmentation을 걸어 train batch가 val batch보다 어려운 문제이기 때문입니다. **이건 overfitting의 정반대 신호** — augmentation이 잘 작동한다는 증거."

`confusion_matrix.png` 보여주며:

> "혼동 행렬입니다. 대각선이 거의 다 채워져 있고 비대각선은 1~2장. 8 클래스 중 4개가 100%, 나머지도 99%대."

`classification_report.txt`와 `metadata.json` 보여주며:

> "scikit-learn classification report는 클래스별 precision/recall/f1. `metadata.json`엔 best_epoch, val_accuracy, class layout이 들어있고 `trained_models.zip`의 일부입니다 — predict.py가 클래스 라벨 매핑에 씁니다."

### 7-7. 데이터 누수 회피 + Transfer 비교

> "정확도 99.8%라 'overfitting/누수 아니냐'고 의심하실 수 있습니다. 핵심은 **augment과 split의 순서**입니다."

> "**잘못된 순서 (augment → split)**: augmented_directory를 그대로 split하면 같은 원본의 변형들 — `image (1).JPG`와 `image (1)_Flip_0.JPG` — 이 train과 val에 흩어집니다. 모델이 train에서 외운 답을 val에서 거의 같은 사진으로 만나서 **가짜 100% 정확도**. 이걸 데이터 누수라고 합니다."

> "**우리 순서 (split → augment)**: 원본 `images/`만 split하고 train batch에만 메모리에서 random augmentation. 변형이 디스크에 안 남으니 val에 노출될 경로가 없음. 누수 0%."

> "**증거**: v1을 augmented_directory로 학습 → 100% (의심). v2로 원본 + online augmentation으로 바꾸니 99.79% (자연스러움)."

**Transfer model 비교 (선택)**:

> "기본은 ScratchCNN 한 개로 99.79% PDF 통과. 비교용으로 EfficientNet-B0 transfer learning도 같이 학습할 수 있게 했습니다."

> "**EfficientNet-B0**는 ImageNet 100만 장(1000 클래스)으로 사전학습된 모델. CNN의 초기·중간 layer가 학습한 '선·곡선·텍스처' 인식 능력은 잎 사진에도 그대로 활용 가능. 마지막 1000-class 분류층만 8-class로 갈아끼우고 추가 학습."

> "**Two-stage fine-tuning**: Stage 1(epoch 1-5) backbone 동결 + classifier만 학습, Stage 2(epoch 6+) 전체 unfreeze + LR 1/10. 우리 결과에서 epoch 5→6 사이 val_loss가 0.05→0.011로 점프하는데 이게 unfreeze 효과입니다."

> "**두 모델이 비슷한 정확도** (99.79% vs 99.86%) → 우리 데이터셋이 명확해서 두 모델 모두 풀 수 있는 거지 한 쪽 모델의 트릭이 아닙니다. 다만 calibration은 ScratchCNN이 더 정직 — 틀릴 때 39~57% confidence로 망설입니다."

### 📂 관련 코드 (이 섹션 전체)

- **`src/leaffliction/models/scratch_cnn.py`** — ScratchCNN. 4 conv block + GAP head.
- **`src/leaffliction/models/transfer.py`** — TransferModel. `efficientnet_b0(pretrained=True)` 가져와 마지막 classifier만 교체. `freeze()`/`unfreeze()` 메서드.
- **`src/leaffliction/trainer.py`** — `train()` 함수. PyTorch 5줄 루프, optimizer/scheduler/early stop, two-stage fine-tune 분기.
- **`src/leaffliction/dataset.py`** — `LeafDataset` PyTorch Dataset class. `__getitem__`에서 transform 적용(online augmentation).
- **`train.py`** — typer wrapper. stratified split, WeightedRandomSampler 구성, model dispatch(scratch/transfer/both), 결과 시각화 + zip + signature.

---

## 8. Part 4 (3/4) — Unit_test1 (Apple)

### 🎯 평가자가 확인하는 것

> "Take the images from the Unit_test1 folder and **give one point for each correct Apple leaf image classified**. Ensure the classification matches the image name and **replace the latter to prevent the student from accessing it**."

→ 평가자가 파일명을 무작위로 바꿔서 `predict.py`가 진짜 사진만 보고 맞추는지 확인.

### 📖 개념 — 이걸 왜 하나

**Generalization test** — 학습/val 분포 안에서의 정확도가 아니라, 실제로 **본 적 없는 새 사진**에 대한 예측 능력 확인. ML 시스템의 진짜 가치는 unseen data 처리.

평가표가 파일명을 바꾼다는 건 학생이 "파일명에서 정답을 읽는" 부정 행위를 막기 위함. 우리 `predict.py`는 metadata.json에서 클래스 라벨만 읽고 **파일명에 의존하지 않습니다** — 안전.

### 💻 명령

```bash
# 폴더 통째로 한 번에 — 다중 모드, 콘솔 표 출력 + 자동 self-check
./predict.py /tmp/test_images/Unit_test1/

# PNG도 같이 저장
./predict.py /tmp/test_images/Unit_test1/ --save /tmp/out_unit1/

# PDF 예시 그대로 단일 모드도 동일 명령으로 지원
./predict.py /tmp/test_images/Unit_test1/Apple_healthy1.JPG
```

### 🎤 대사

> "Unit_test1은 Apple 4 클래스에서 뽑은 10장입니다. `predict.py`에 폴더를 넘기면 자동으로 `*.JPG`를 수집해 한 번에 예측. 단일 이미지로 호출하면 PDF 예시처럼 figure를 띄우고, 폴더면 콘솔 표로 출력합니다."

다중 모드 출력 예시:

```
Predicting 10 images with model=scratch...
  OK   Apple_healthy      (99.8%)  ← Apple_healthy1.JPG
  OK   Apple_Black_rot    (99.1%)  ← Apple_BlackRot2.JPG
  ...
Self-check: 10/10 = 100.00%
```

> "파일명이 클래스명으로 시작하면 자동 self-check이 동작합니다. 평가자가 파일명을 무작위로 바꿔도 빈 칸으로 표시될 뿐 예측은 그대로 동작합니다 — **파일명은 예측 입력에 안 쓰입니다**, self-check 보조 정보일 뿐."

### 📂 관련 코드

- **`predict.py`** — typer CLI. 인자 개수/디렉토리 여부로 단일/다중 모드 자동 dispatch.
  - `_expand(paths)` — 디렉토리는 `rglob("*.JPG")`로 펼침.
  - `_guess_class_from_name(name, classes)` — 파일명 prefix 매칭으로 self-check.
  - 다중 모드일 때 `--save dir/`이면 `<stem>_pred.png`로 일괄 저장.
- **`src/leaffliction/predictor.py`**
  - `load_artifact(zip, prefer)` — zip 한 번만 unzip하고 모델 + classes 로드(다중 모드에서 모델 1회만 load).
  - `predict_one(artifact, image)` — 한 장 처리: PIL load → resize → normalize → forward → softmax.
  - `predict_many(...)` — `load_artifact` 한 번 + `predict_one` 반복.
  - `render(result, save)` — 2 패널 figure(원본 + mask transform) + suptitle(class + confidence).

### ⚠️ 함정

- 10장 다 맞으면 5점.
- 평가자가 파일명을 바꿨다면 self-check는 자동으로 skip → 콘솔 결과를 평가자가 직접 채점.

---

## 9. Part 4 (4/4) — Unit_test2 (Grape)

### 🎯 평가자가 확인하는 것

> "Take the images from the Unit_test2 folder. **If the 10 images are misclassified, ask yourself how the student was able to get a good classification in his validation set.**"

→ 10장 다 틀리면 누수 의심 명시.

### 📖 개념 — 이걸 왜 하나

`Unit_test2`는 Grape 4 클래스. 핵심은 **모델이 단일 식물(Apple)에 overfit한 게 아니라 Grape에도 일반화**되는지 확인. 평가표가 명시적으로 누수 의심을 언급한 단계라 우리 split→augment 순서의 정당성을 입증할 기회.

### 💻 명령

```bash
./predict.py /tmp/test_images/Unit_test2/
./predict.py /tmp/test_images/Unit_test2/ --save /tmp/out_unit2/
```

### 🎤 대사

> "Unit_test2는 Grape 4 클래스에서 뽑은 10장입니다. 같은 폴더 다중 모드로 한 번에 돌립니다."

> "평가표가 명시적으로 'Unit_test2 다 틀리면 누수 의심하라'고 적었습니다. §7에서 설명한 split→augment 순서가 누수를 차단했고, val accuracy 99.79%가 자연스러운 결과라는 걸 Unit_test2도 통과하는 것으로 보여드립니다."

### ⚠️ 함정

- 10장 모두 틀림 → 누수 의심 명시. §7 누수 회피 설명을 한 번 더 강조.
- 일부 틀려도 PlantVillage 분포 안이면 자연스러운 오답. confusion_matrix와 일치하는 패턴인지 확인.

---

## 10. 자주 받는 질문 (Q&A)

### Q1. 왜 CNN을 직접 만들었어요? Transfer learning이 더 쉽지 않아요?

> "둘 다 만들었습니다. Default는 ScratchCNN. 이유는 (1) 모든 레이어를 직접 설계해서 한 줄씩 설명할 수 있다는 게 디펜스에 안전하고 (2) 평가표가 '모델 설명' 항목에 5점을 배정해서 black-box 사전학습 모델보다 직접 만든 모델이 설명 면에서 유리합니다."

### Q2. 왜 EfficientNet-B0이에요?

> "비슷한 정확도일 때 EfficientNet-B0이 파라미터/연산량 효율이 좋아 modern 표준 baseline. ResNet-50이 25M 파라미터인데 B0은 5M으로 1/5. CPU 추론에서도 빠릅니다."

### Q3. 데이터 누수가 정확히 뭐예요?

> "누수는 모델이 평가 단계에서 보면 안 되는 정보를 학습 중에 이미 본 상태. 가장 흔한 형태가 augment-then-split. 같은 원본의 변형들이 train과 val에 흩어지면 모델이 train에서 외운 답을 val에서 거의 그대로 만나서 가짜 100% 정확도. 우리는 원본만 split + train batch에 메모리에서 online augmentation으로 차단."

### Q4. Online augmentation이 정확히 뭐예요?

> "디스크에 augmented 이미지를 안 만들고, 학습 중 batch를 만들 때마다 메모리에서 random 변형을 적용. `LeafDataset.__getitem__`에서 transform이 매 호출마다 새 random으로 실행. 같은 image (1)이 epoch마다 다른 변형으로 보이니까 사실상 무한 데이터 효과 + 디스크에 안 남아 val 노출 경로 0."

### Q5. WeightedRandomSampler가 뭐예요? augmented_directory로 균형 맞추면 되잖아요?

> "augmented_directory는 Part 2의 PDF 요구사항을 위해 만들지만 학습엔 안 씁니다(누수 때문). 그래서 클래스 불균형을 보정할 다른 방법이 필요하고 그게 `WeightedRandomSampler`. 각 sample의 weight = 1 / 그 클래스 크기로 줘서 batch마다 작은 클래스가 자주 뽑힘. 디스크 저장 없이 같은 효과 + 누수 회피."

### Q6. Stratified split이 일반 random split이랑 어떻게 달라요?

> "일반 random split은 운나쁘게 val에 특정 클래스가 0장일 수도 있어 측정 불가. Stratified는 클래스 비율 유지하면서 8:2로 나눠 모든 클래스가 val에 일정 비율 들어가게 보장. `sklearn.train_test_split(stratify=labels)` 한 줄."

### Q7. seed=42는 왜 42예요?

> "Douglas Adams의 '은하수를 여행하는 히치하이커를 위한 안내서'에서 '삶, 우주, 모든 것의 답'이 42라서 ML 커뮤니티 농담 같은 표준이 됐습니다. 숫자 자체는 의미 없고 재현성 위해 고정값이면 충분."

### Q8. 정확도 99%는 너무 좋은데 진짜인가요?

> "세 가지 증거: (1) PlantVillage는 통제된 회색 배경 + 클래스 간 시각 차이 명확. 학계 논문에서도 EfficientNet/ResNet으로 95-99% 흔함. (2) ScratchCNN과 EfficientNet 두 다른 방식이 0.07pp 이내 → 데이터셋 명확. (3) confusion matrix 자연스러움(1-2장 misclassified 흩어짐). 추가로 `eval_val.py`로 즉시 재검증 가능."

### Q9. Overfitting은 어떻게 막았나요?

> "다섯 가지 동시 사용: (1) Dropout(0.4), (2) weight_decay=1e-4, (3) online augmentation, (4) early stopping patience=5, (5) ReduceLROnPlateau. `learning_curves.png`에서 train과 val이 함께 수렴, 종종 train_loss > val_loss인 epoch도 있어요(augmentation이 train을 더 어렵게)."

### Q10. 신뢰도가 100.0%로 뜨는 건 무슨 의미예요?

> "softmax 후 가장 큰 클래스 확률. PlantVillage가 통제된 환경이라 모델이 매우 확신하기 쉽고, 신경망은 학습 잘 되면 과잉 자신감(overconfidence) 경향이 있어 softmax exponential로 1.0에 거의 붙어버립니다. 일반 야외 사진이면 보통 70~95%로 떨어집니다."

### Q11. uv는 왜 쓰나요?

> "pip + virtualenv + pyenv + pip-tools를 통합한 modern Python 도구. Rust 작성, pip 대비 10-100배 빠름, `uv.lock`으로 재현성. 2024-2026 Python 표준으로 자리잡는 중."

### Q12. 코드 어디 있어요?

```
루트 entrypoints: Distribution / Augmentation / Transformation / train / predict.py
실제 로직:       src/leaffliction/*.py
모델:            src/leaffliction/models/{scratch_cnn, transfer}.py
학습 루프:        src/leaffliction/trainer.py
추론:            src/leaffliction/predictor.py
데이터셋:         src/leaffliction/dataset.py
변환:            src/leaffliction/transform.py
증강:            src/leaffliction/augment.py
시각화:           src/leaffliction/viz.py
시그니처:         src/leaffliction/signature.py
테스트:           tests/test_*.py  (28개 pytest)
스크립트:         scripts/eval_val.py / verify.sh / check_no_dataset.sh
디자인 문서:      docs/superpowers/specs/, plans/
```

### Q13. 모르는 질문이 나오면?

> "솔직히 '그건 모르겠는데 코드 보면서 같이 보겠습니다' + 코드 띄움. 평가표 0점 조건은 '설명 못함'이지 '한두 개 모름'이 아님. 정직성이 더 안전."

---

## 11. 위험 시나리오 + 대응

### A. `make verify` 실패 (signature 불일치)

```bash
ls -la trained_models.zip augmented_directory.zip
shasum trained_models.zip augmented_directory.zip
# → 백업 USB 시도 → 그래도 안 되면 재학습(~50분)
```

### B. `uv sync` 실패 (plantcv 빌드)

```bash
uv pip install plantcv --no-build-isolation
# 또는
python -m pip install -e .
```

### C. matplotlib 창 안 뜨는 환경 (SSH/Docker)

```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png
```

### D. "Unit_test에서 다 틀렸어요"

→ §7 누수 회피 설명. `confusion_matrix.png`로 내부 분포 정확도 보여주기. PlantVillage 외 데이터에선 자연스럽게 낮을 수 있음 인정.

### E. `./Distribution.py: command not found`

```bash
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py
# 또는
source .venv/bin/activate
# 또는
uv run python Distribution.py ./images
```

### F. 평가자가 다른 dataset path를 줌

→ 모든 entrypoint가 path 인자를 받음. 우리 `images/`에 종속 안 됨.

```bash
./Distribution.py /평가자/제공/경로
./predict.py "/평가자/제공/Unit_test1/"
```

### G. 신뢰도 100%가 의심스럽다는 질문

→ Q10 답. softmax overconfidence는 알려진 현상. confusion matrix와 misclassified 표(§6)로 정직한 증거 제시.

---

## 12. USB / D-1 사전 체크리스트

평가 전날:
- [ ] `make lint` 통과
- [ ] `make test` 통과 (pytest 28개)
- [ ] `make verify` 통과
- [ ] `./scripts/eval_val.py images` ≥ 90% (99.79% 확인)
- [ ] `./scripts/eval_val.py images --model transfer` ≥ 90% (99.86% 확인)
- [ ] `git status`에 `*.zip / *.pt / images/ / augmented_directory/` 없음

USB:
- [ ] `trained_models.zip` (~20MB)
- [ ] `augmented_directory.zip` (~187MB)
- [ ] 백업 USB 또는 cloud (Google Drive, AirDrop, scp)
- [ ] 노트북 충전기

평가 시작 5분 전:
- [ ] `cd ~/42/Leaffliction && source .venv/bin/activate`
- [ ] `artifacts/learning_curves.png` 미리 한 번 열어 확인
- [ ] `artifacts/confusion_matrix.png` 미리 한 번 열어 확인
- [ ] `artifacts/classification_report.txt` 미리 한 번 열어 확인

---

## 13. 마무리 한 줄 pitch (Conclusion 칸)

> "PDF의 5개 entrypoint와 평가표의 0점 함정(augmented_directory 균형, signature.txt 일치, 100+ 이미지 ≥90%)을 모두 통과 + 데이터 누수를 회피한 정직한 결과입니다. ScratchCNN을 처음부터 설계해 모든 레이어를 설명할 수 있고, EfficientNet-B0 transfer model도 비교용으로 두어 CNN의 production 활용 방식까지 함께 시연 가능합니다."

---

## 🔗 관련 문서

- 디자인 결정 + 근거: [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)
- 구현 plan: [docs/superpowers/plans/2026-04-28-leaffliction.md](superpowers/plans/2026-04-28-leaffliction.md)
- 산출물 무결성: `signature.txt` + `make verify`
