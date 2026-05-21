# Leaffliction — Defense Script (대본)

> 평가표(Intra Projects Leaffliction Edit) 순서를 그대로 따라가며 명령을 실행하고 개념을 설명할 수 있게 만든 대본.
>
> 각 단계 구조:
> - **🎯 평가자가 확인하는 것** — 평가표 항목 그대로
> - **💻 명령** — 그대로 복붙
> - **🎤 대사** — 인용 박스 안의 문장을 입으로 읽으면 됨 (한국어 / 영어 둘 다 적어둠)
> - **⚠️ 함정** — 0점 위험 또는 자주 묻는 후속 질문

---

## 평가표 흐름 (한눈에)

```
Error Management   ─ signature.txt diff + flake8 norm
   ↓
Part 1             ─ ./Distribution.py ./Apple
   ↓
Part 2             ─ ./Augmentation.py ./Apple/apple_healthy/image (1).JPG
   ↓
Part 1 (verify)    ─ ./Distribution.py ./augmented_directory  [0점 함정]
   ↓
Part 3             ─ ./Transformation.py ./Apple/apple_healthy/image (1).JPG
   ↓
Part 4 (1) 정확도  ─ test set 100+ 이미지 ≥90%
Part 4 (2) 설명    ─ ML 모델 설명 (0~5점, 가장 큰 단일 점수)
Part 4 (3) Unit_test1  ─ predict 10개 Apple 이미지
Part 4 (4) Unit_test2  ─ predict 10개 Grape 이미지
```

---

## 0. 시작 전 환경 세팅

평가 시작 5분 전에 평가자 컴퓨터(또는 본인 노트북)에서 미리.

```bash
git clone https://github.com/keonwoo98/Leaffliction.git
cd Leaffliction
uv sync                              # 의존성 설치 (~30초)
source .venv/bin/activate
```

USB에서 데이터셋과 학습 zip 복사:

```bash
cp /Volumes/USB/trained_models.zip .
cp /Volumes/USB/augmented_directory.zip .
unzip -q augmented_directory.zip     # Part 1 verify 단계용
unzip -q leaves.zip                  # 평가표 첨부 leaves.zip → images/ 로 풀려야 함
```

> 평가표 attachments에 `leaves.zip`이 있음. 우리 `images/` 폴더 구조와 동일해야 함.
> `leaves.zip` 안에 `Apple/`, `Grape/` 같은 상위 폴더가 있다면 그대로 사용하면 되고, 우리가 `images/`로 풀어놨다면 평가자에게 "동일 데이터셋입니다, 폴더명만 다릅니다" 설명.

---

## 1. Error Management

### 🎯 평가자가 확인하는 것

- `signature.txt` 안의 해시가 `.zip` 데이터셋의 실제 해시와 동일한지 `diff`
- Python인 경우 `flake8`으로 norm 검사

### 💻 명령

```bash
# Signature 검증
shasum trained_models.zip augmented_directory.zip
cat signature.txt
diff <(shasum trained_models.zip augmented_directory.zip) signature.txt
# (또는 우리가 만든 단축 명령)
make verify

# Python norm 검사 (평가표가 flake8을 norminette로 alias)
flake8 src tests *.py
```

### 🎤 대사

> "먼저 signature 검증부터 하겠습니다. PDF Chapter V에서 dataset zip의 SHA1 해시를 `signature.txt`에 적어두고 평가일에 zip 실제 해시와 비교하라고 명시했고, 일치하지 않으면 0점입니다. `make verify`가 두 zip의 `shasum` 결과를 `signature.txt`와 `diff`하는 단축 명령입니다."

명령 실행 후 출력이 깨끗하면(차이가 없으면):

> "두 줄 다 일치합니다. zip이 학습 당시 그대로라는 증거입니다."

그 다음 `flake8`:

> "다음은 Python norm입니다. 평가표에서 `flake8`을 `norminette_python` alias로 정의했습니다. 프로젝트 루트 + `src/` + `tests/`를 검사합니다."

### ⚠️ 함정

- **signature 불일치 → 평가 즉시 종료, 0점**. USB에서 zip 다시 복사하거나 백업 USB로 재시도.
- `flake8` 에러 1개라도 → **Norme flag** 가능. 우리는 사전에 `make lint` 통과 확인됨.

---

## 2. Part 1 — Analysis of the Data Set

### 🎯 평가자가 확인하는 것

> "Read the code, Run the code and pie chart as in the subject must appear."

코드 읽기 + 실행 + subject PDF에 있던 pie chart 형태가 떠야 함.

### 💻 명령

```bash
./Distribution.py ./images/Apple
```

또는 전체 데이터셋(8 클래스 한 번에):

```bash
./Distribution.py ./images
```

### 🎤 대사

명령 실행 직전:

> "Part 1은 데이터셋 분석입니다. `Distribution.py`는 각 하위 폴더를 하나의 클래스로 보고 이미지 개수를 세서 pie chart와 bar chart 두 개로 시각화합니다."

차트 뜨면:

> "Apple 4 클래스만 본 결과입니다. `Apple_healthy`가 1640장으로 가장 많고 `Apple_rust`가 275장으로 가장 적습니다. 약 6배 불균형이고, 이게 Part 2 data augmentation의 동기입니다."

코드 한 번 보여주기 (`src/leaffliction/dataset.py`):

> "구현은 `dataset.py`의 `discover_classes` 함수입니다. `pathlib.Path.rglob`으로 모든 `*.JPG`를 찾고 클래스(부모 폴더 이름)별로 그룹핑합니다. matplotlib + seaborn으로 pie + bar를 한 figure에 출력합니다."

### ⚠️ 함정

- pie chart는 subject PDF의 예시와 시각적으로 동일한 형태여야 함 → 우리는 일치.
- bar chart 위 막대마다 숫자 라벨이 있어서 정확한 개수를 즉시 볼 수 있음 (가독성 가점 포인트).

---

## 3. Part 2 — Data Augmentation

### 🎯 평가자가 확인하는 것

> "Read the code, Run the code and 6 images as in the subject must appear. 6 versions of the same image must have been created. ls: image (1)_Flip.JPG, _Rotate.JPG, _Skew.JPG, _Shear.JPG, _Crop.JPG, _Distortion.JPG"

### 💻 명령

```bash
./Augmentation.py "images/Apple_healthy/image (1).JPG"
ls "images/Apple_healthy/image (1)"*
```

### 🎤 대사

> "Part 2는 6가지 augmentation입니다. PDF가 명시한 6개는 Flip, Rotate, Skew, Shear, Crop, Distortion이고 정확히 그 이름으로 파일을 저장합니다."

명령 실행 후 `ls` 결과 보여주며:

> "원본 옆에 `image (1)_Flip.JPG`처럼 suffix만 붙여서 6개가 생성됐습니다. PDF가 보여준 파일명 규칙과 정확히 일치합니다."

각 변환이 뭔지 물어볼 가능성 있음 — 코드 보여주기 (`src/leaffliction/augment.py`):

> "구현은 Albumentations 라이브러리입니다. 6개를 dict로 정의했습니다:
> - **Flip** = HorizontalFlip — 좌우 반전
> - **Rotate** = Rotate(±30°) — 회전
> - **Skew** = Affine(shear x+y, ±15°) — x·y 양방향 평행사변형 변형
> - **Shear** = Affine(shear x, ±25°) — x축만 더 강한 평행사변형
> - **Crop** = RandomResizedCrop(70~100%) — 부분 잘라 다시 256×256으로 확대
> - **Distortion** = OpticalDistortion(±0.4) — 렌즈 왜곡 흉내"

시연 끝나면 정리:

```bash
rm "images/Apple_healthy/image (1)_"*.JPG
```

> "다음 단계 영향 안 주게 sibling 6개는 정리하겠습니다."

### ⚠️ 함정

- 파일명이 평가표 예시와 일치해야 함. 우리는 `_Flip`, `_Rotate` 등 대문자 시작 일치.
- "원본은 안 지웠죠?" 물어보면 → 원본은 그대로 두고 6개 sibling만 추가.

---

## 4. Part 1 추가 검증 — augmented_directory 균형 ⚠️ (0점 함정)

### 🎯 평가자가 확인하는 것

> "Now you have to use the program of the part 1 on the augmented_directory of the student … each part of the pie chart must be equal. **If the pie chart is still the same as above you must put 0** to this exercise and exercise Part 1 Analysis of the Data Set."

→ **augmented_directory가 균형 잡혀 있지 않으면 Part 1, Part 2 둘 다 0점**.

### 💻 명령

```bash
./Distribution.py ./augmented_directory
```

(만약 `augmented_directory/`를 unzip 안 했다면 먼저:)

```bash
unzip -q augmented_directory.zip
./Distribution.py ./augmented_directory
```

### 🎤 대사

명령 실행 직전:

> "이제 같은 `Distribution.py`를 augmented 폴더에 실행합니다. 평가표가 명시적으로 경고하는 부분입니다. augmented_directory가 원본과 동일하게 불균형하면 Part 1, Part 2 둘 다 0점이라고 명시돼 있습니다."

차트 뜨면 (각 조각이 12.5%, 8 × 1640 = 13,120장):

> "보시다시피 8 클래스 모두 1640장으로 동일합니다. pie chart 8 조각이 정확히 균등합니다. 가장 많았던 `Apple_healthy`(1640)에 맞춰서 나머지 7 클래스를 augmentation으로 1640까지 채웠습니다."

구현 위치 (`src/leaffliction/augment.py:balance_directory`):

> "`balance_directory` 함수에서 (1) 원본을 복사하고, (2) 부족한 만큼 6개 op 중 random하게 골라 적용해 채워넣고, (3) 마지막에 `augmented_directory.zip`을 만듭니다. zip은 PDF Chapter V의 signature 요구사항 때문입니다."

### ⚠️ 함정

- 이 단계에서 처음 pie chart와 동일하게 보이면 → Part 1과 Part 2 동시 0점. 절대 안전해야 함.
- 평가자가 `augmented_directory` 위치를 물으면 "현재 디렉토리에 unzip 했습니다"로 답.

---

## 5. Part 3 — Image Transformation

### 🎯 평가자가 확인하는 것

> "Read the code, Run the code and 6 images as in the subject must appear. The techniques used must be able to extract the characteristics of the plants, you can ask for explanations on each one of the transformations."

### 💻 명령

```bash
./Transformation.py "images/Apple_healthy/image (1).JPG"
```

옵션 확인용:

```bash
./Transformation.py -h
```

배치 모드 시연 (선택):

```bash
./Transformation.py -src ./images/Apple_healthy -dst /tmp/transformed -mask
```

### 🎤 대사

명령 실행 직전:

> "Part 3은 plantCV로 잎의 특징을 시각화하는 단계입니다. 6가지 변환을 한 figure 위쪽에, 9채널 color histogram을 아래쪽에 배치했습니다."

화면 뜨면 위쪽 6개 패널 하나씩 가리키며:

> 1. **Original** — 원본 RGB
> 2. **Gaussian blur** — 작은 노이즈 완화. mask가 깔끔해지는 효과.
> 3. **Mask** — 잎과 배경 분리. LAB 색공간에서 chroma magnitude를 계산하고 Otsu threshold + 가장 큰 connected component + fill holes를 적용했습니다.
> 4. **ROI** — 관심 영역 박스 + mask overlay.
> 5. **Analyze object** — 잎 모양 분석 (면적, 둘레, 중심 등 metric 계산).
> 6. **Pseudolandmarks** — 잎 가장자리 따라 자동 landmark.

mask가 왜 robust한지 물어볼 가능성 큼 — 코드 보여주기 (`src/leaffliction/transform.py:_binary_mask`):

> "초창기엔 HSV saturation threshold를 썼는데 healthy 사과 잎(회색-녹색)에서 잘 안 잡혀서 LAB chroma magnitude로 바꿨습니다. LAB는 a축 = 녹-적, b축 = 청-황. 회색 배경은 a≈128, b≈128이라 chroma `sqrt((a-128)² + (b-128)²)`가 0에 가깝고, 녹색 잎과 갈색 병변은 둘 다 chroma가 큽니다. 그 뒤 Otsu로 자동 threshold, morphological opening으로 노이즈 제거, 가장 큰 connected component만 유지, scipy `binary_fill_holes`로 잎 안쪽 구멍을 메웁니다."

9채널 histogram 가리키며:

> "아래는 9채널 color histogram입니다. RGB 3채널, HSV 3채널, LAB 3채널을 같이 그려서 클래스별 색 분포 특성을 비교할 수 있게 했습니다."

### ⚠️ 함정

- plantCV의 default thresholding 함수만 쓰면 healthy 잎에서 mask 절반밖에 안 잡힘 → 우리는 LAB chroma로 우회.
- 평가자가 "왜 plantcv를 그대로 안 썼냐"고 물으면 → "plantcv `threshold.binary`는 단일 채널 입력을 받는데 어떤 채널을 쓸지가 핵심 문제였고, healthy 잎은 saturation으로 안 잡혀서 LAB chroma를 직접 만들어 넣었습니다"라고 답.

---

## 6. Part 4 (1/4) — Classification 정확도 (≥90%)

### 🎯 평가자가 확인하는 것

> "Ask the student to run his program on a test set of minimum 100 images, the result of good prediction must be over 90%."

### 💻 명령

```bash
./scripts/eval_val.py ./images
```

옵션으로 transfer 모델도:

```bash
./scripts/eval_val.py ./images --model transfer
```

### 🎤 대사

명령 실행 직전:

> "Part 4 첫 번째 항목은 100장 이상에서 90% 이상 정확도입니다. `eval_val.py`라는 스크립트를 만들었는데, train.py가 학습 시 `random_state=42`로 stratified 80/20 split을 했기 때문에 같은 seed로 split을 재현하면 학습에 안 쓴 1,445장(20%)을 정확히 다시 뽑을 수 있습니다. 그 1,445장을 forward만 돌려서 정확도를 계산합니다."

결과 출력되면:

> "1442/1445, 99.79%입니다. PDF 요구 90%를 약 10pp 상회합니다. 8 클래스 per-class breakdown도 같이 출력되는데 모든 클래스가 99% 이상입니다."

transfer도 보여주기:

> "비교용 transfer 모델은 99.86%로 약간 더 높습니다. ScratchCNN과 0.07pp 차이라 데이터셋이 명확해서 두 모델 모두 잘 풀린 거지 한 쪽 모델의 트릭이 아닙니다."

### ⚠️ 함정

- "이거 train accuracy 아니냐?" → 아니, `seed=42`로 split을 재현해서 train.py가 학습 중 본 적 없는 1,445장입니다. `train.py`의 출력값을 안 믿어도 됩니다.
- "100장만 보자"고 하면 → 1,445장이 100장보다 많으니 자동 충족. 원하면 `--max 100`도 추가 가능하지만 1,445가 더 강한 증거.

---

## 7. Part 4 (2/4) — 모델 설명 (5점, 가장 중요)

### 🎯 평가자가 확인하는 것

> "The student must be able to explain the machine learning model he has chosen and how it works. 0 if can't explain, 5 if explanations are fluid."

→ **단일 항목에서 5점**. 모든 항목 중 가장 큰 점수 항목. 여기서 막히면 큰 손실.

설명 순서: (1) 큰 그림 → (2) 데이터 흐름 → (3) 모델 구조 → (4) 학습 루프 → (5) overfitting/누수 방지.

### 💻 보조 명령 (대사 중간에 코드 띄우기)

```bash
# 모델 구조 코드
cat src/leaffliction/models/scratch_cnn.py

# 학습 루프
cat src/leaffliction/trainer.py | head -80

# 결과 시각화
open artifacts/learning_curves.png
open artifacts/confusion_matrix.png
cat artifacts/classification_report.txt
cat artifacts/metadata.json
```

### 🎤 대사 (전체 통째로 읽으면 5점)

#### (1) 왜 CNN인가

> "이건 8 클래스 leaf disease 분류 문제고 입력은 256×256 RGB 사진입니다. 사진 분류는 **CNN(Convolutional Neural Network)**이 표준입니다. CNN은 작은 필터로 이미지를 훑어서 패턴을 찾고, 여러 layer를 통과시키며 점점 추상적인 의미를 추출하는 신경망입니다."

> "직관적으로는 사람이 그림을 볼 때처럼 4단계로 추상화합니다. 처음엔 선·점 → 그 다음 모서리·텍스처 → 그 다음 잎맥·반점 → 마지막으로 잎의 종류·질병. 우리 CNN의 4개 conv block이 이 4단계를 자동 학습합니다."

#### (2) 데이터 흐름 (3단계)

> "데이터는 세 단계를 거칩니다."

> "첫째, **stratified 80/20 split**입니다. `sklearn.train_test_split`에 `stratify=labels`와 `random_state=42`를 줘서 클래스 비율을 유지하면서 동일하게 재현 가능한 분할을 합니다. Apple_healthy 1640장이면 train 1312장, val 328장 이런 식으로 모든 클래스가 8:2를 지킵니다."

> "둘째, train set 내부에서도 클래스 불균형이 남아 있으니 **WeightedRandomSampler**를 씁니다. 각 sample의 weight = 1 / 그 클래스 크기. Apple_rust 220장의 한 sample은 weight 1/220, Apple_healthy 1312의 한 sample은 1/1312. 결과적으로 batch마다 작은 클래스가 자주 뽑혀서 클래스 균등 batch가 됩니다."

> "셋째, train batch에만 **online augmentation**을 적용합니다. `transforms.v2.RandomHorizontalFlip(p=0.5)`과 `RandomRotation(15°)`. 메모리에서 batch 만들 때마다 random하게 변형되니까 같은 이미지가 epoch마다 다르게 보이고, 디스크엔 저장 안 됩니다. **val set엔 augmentation 적용 안 함** — 측정 정직성을 위해."

#### (3) 모델 구조 — ScratchCNN

`src/leaffliction/models/scratch_cnn.py` 화면에 띄워둔 채:

> "모델은 두 부분입니다. `self.features`(특징 추출, conv block 4개)와 `self.head`(분류기, GAP + Dropout + Linear)."

> "한 conv block은 `_conv_block` 함수에 정의돼 있고 7층입니다: Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2)."

> "**Conv2d(in_ch=3, out_ch=32, kernel_size=3)**은 3×3 짜리 작은 필터 32개로 입력을 훑어서 32개의 새 채널을 만듭니다. 채널 개념이 헷갈릴 수 있는데, 채널 1개 = 같은 사진을 한 가지 관점에서 본 흑백 지도라고 보면 됩니다. 처음엔 R, G, B 3개. Conv 통과하면 32 → 64 → 128 → 256으로 늘어나고 각 채널이 다른 종류의 패턴 지도가 됩니다."

> "**BatchNorm2d**는 출력을 평균 0, 분산 1로 정규화해서 학습을 안정시킵니다. **ReLU**는 음수→0, 양수→그대로, 신경망에 비선형성을 부여합니다. ReLU가 없으면 layer를 아무리 쌓아도 결국 하나의 선형 모델이 됩니다."

> "**MaxPool2d(2)**는 2×2 영역에서 최대값만 남깁니다. 공간 해상도가 절반으로 줄고 (256→128→64→32→16), 작은 디테일은 버리고 큰 그림에 집중하게 됩니다."

> "block을 4번 거치면 (3, 256, 256)이 (256, 16, 16)이 됩니다. 채널이 늘어나면서 공간이 줄어드는, CNN의 전형적인 깔때기 구조입니다."

> "Head는 4층입니다. `AdaptiveAvgPool2d(1)`은 GAP(Global Average Pooling). (256, 16, 16)을 (256, 1, 1)로 만들어서 위치 정보를 평균 내고 '사진 전체에 256개 패턴이 얼마나 강한가'라는 256차원 벡터로 압축합니다. 그 다음 Dropout(0.4)로 40% 뉴런을 학습 중 끄고, 마지막 Linear(256, 8)로 8 클래스 logit을 만듭니다."

> "총 파라미터는 약 1.18M입니다."

#### (4) 학습 루프 — PyTorch 5줄

`src/leaffliction/trainer.py` 화면에 띄운 채:

> "학습 루프의 핵심은 PyTorch 표준 5줄입니다:"

```python
logits = model(x)              # ① forward — 예측
loss = criterion(logits, y)    # ② loss — 정답과 차이
optimizer.zero_grad()          # ③ 이전 gradient 지움
loss.backward()                # ④ backward — chain rule로 미분 자동 계산
optimizer.step()               # ⑤ weight 살짝 업데이트
```

> "forward에서 모델이 예측을 만들고, criterion이 정답과의 차이를 loss로 계산하고, `loss.backward()` 한 줄이 chain rule을 자동 적용해서 수백만 weight의 미분을 한 번에 계산합니다. `optimizer.step()`이 그 반대 방향으로 weight를 살짝 옮깁니다. 이걸 batch 단위로 25 epoch × 약 180 step = 4,500번 정도 반복합니다."

> "Loss는 `CrossEntropyLoss` — 분류의 표준입니다. Optimizer는 `Adam(lr=1e-3, weight_decay=1e-4)`. Adam은 weight마다 step 크기를 적응적으로 조절해서 SGD보다 빠르게 수렴합니다. weight_decay는 L2 regularization으로 overfitting 완화."

> "추가로 `ReduceLROnPlateau(factor=0.5, patience=2)` scheduler — val_accuracy가 2 epoch 정체되면 learning rate를 절반으로 낮춰서 후반 미세조정. 그리고 early stopping(`patience=5`)로 val이 5 epoch 동안 개선 없으면 학습 종료, overfitting 진입 전에 차단합니다."

#### (5) 결과 시각화

`learning_curves.png` 보여주며:

> "학습 곡선입니다. train_loss와 val_loss가 함께 감소하고 마지막에 수렴합니다. 어떤 epoch는 train_loss > val_loss인 경우도 있는데, 이건 train에 augmentation을 걸어서 train batch가 val batch보다 어려운 문제이기 때문입니다. **overfitting의 정반대 신호**입니다."

`confusion_matrix.png` 보여주며:

> "혼동 행렬입니다. 대각선이 거의 다 채워져 있고 비대각선은 1~2장 수준입니다. 8 클래스 중 4개가 100%이고 나머지 4개도 99%대."

`classification_report.txt`와 `metadata.json` 보여주며:

> "scikit-learn classification report는 precision, recall, f1을 클래스별로. metadata.json엔 best_epoch, val_accuracy, class layout이 들어 있고 이게 trained_models.zip의 일부입니다."

#### (6) 데이터 누수 회피 (가장 자주 묻는 후속 질문)

> "정확도가 99.8%라서 'overfitting 아니냐'고 의심하실 수 있습니다. 핵심은 **augment과 split의 순서**입니다. 만약 `augmented_directory/`를 직접 split하면 같은 원본의 변형들(image (1)과 image (1)_Flip_0)이 train과 val에 흩어져서 모델이 외운 답을 그대로 인정받는 가짜 100%가 나옵니다. 이걸 데이터 누수라고 합니다."

> "우리는 원본 `images/`만 split하고 train batch에만 메모리에서 random augmentation을 겁니다. 변형이 디스크에 저장 안 되니 val에 노출될 경로가 없습니다. v1을 augmented_directory로 학습했을 때 100%가 나왔고, v2로 원본+online augmentation으로 바꾸니 99.79%로 자연스럽게 내려왔습니다."

#### (7) 두 모델을 둔 이유 (선택)

> "기본은 ScratchCNN 한 개로도 99.79%로 PDF 통과합니다. 다만 비교용으로 EfficientNet-B0 transfer learning도 같이 학습할 수 있게 해놨습니다. `--model both` 옵션. Transfer는 ImageNet 100만 장으로 사전학습된 가중치를 가져와서 마지막 1000-class 분류층만 8-class로 갈아끼우고, 2단계로 fine-tune합니다. Stage 1(epoch 1-5)은 backbone 동결, Stage 2(epoch 6+)는 전체 unfreeze + LR 1/10로 미세조정. 두 모델 결과가 비슷한 것 자체가 우리 데이터셋이 명확하다는 추가 증거입니다."

### ⚠️ 후속 질문 대비

이 섹션 후 자주 나오는 질문은 §10 Q&A로 분리.

---

## 8. Part 4 (3/4) — Unit_test1 (Apple)

### 🎯 평가자가 확인하는 것

> "Take the images from the Unit_test1 folder and give one point for each correct Apple leaf image classified. Ensure that the classification matches the image name and **replace the latter to prevent the student from accessing it**."

→ 평가자가 파일명을 무작위로 바꿔서 `predict.py`가 진짜 사진만 보고 맞추는지 확인.

### 💻 명령

```bash
# 평가자가 test_images.zip을 풀어둔 위치에서
for img in /tmp/test_images/Unit_test1/*.JPG; do
  echo "=== $img ==="
  ./predict.py "$img"
done
```

또는 한 장씩:

```bash
./predict.py /tmp/test_images/Unit_test1/Apple_healthy1.JPG
```

### 🎤 대사

> "Unit_test1은 Apple 4 클래스에서 뽑은 10장입니다. 평가자가 파일명을 안 보이게 바꾼다고 가정하고, `predict.py`는 사진만 보고 예측합니다."

`predict.py` 출력은 2-패널 figure (원본 + mask transform) + 클래스명 + 신뢰도:

> "각 예측은 클래스명과 신뢰도를 출력합니다. 신뢰도는 softmax 후 가장 큰 확률값."

코드 위치 (`src/leaffliction/predictor.py`):

> "predict는 (1) `trained_models.zip`을 임시 폴더에 풀고 (2) metadata.json에서 클래스 라벨 읽고 (3) `model_scratch.pt` 로드 (default), `--model transfer`로 transfer 모델도 선택 가능 (4) 이미지를 256×256 + ImageNet mean/std normalize (5) forward → argmax."

### ⚠️ 함정

- 10장 모두 맞으면 5점.
- 1~2장 틀려도 PlantVillage 분포 안에 있는 사진이면 confusion matrix와 일치하는 패턴인지 확인. 자연스러운 오답이면 점수 영향 적음.

---

## 9. Part 4 (4/4) — Unit_test2 (Grape)

### 🎯 평가자가 확인하는 것

> "Take the images from the Unit_test2 folder and give one point for each correct Grape leaf image classified. **If the 10 images are misclassified, ask yourself how the student was able to get a good classification in his validation set.**"

→ 10장 다 틀리면 학습 시 누수가 있었던 거 아니냐고 의심하라는 명시 경고.

### 💻 명령

```bash
for img in /tmp/test_images/Unit_test2/*.JPG; do
  echo "=== $img ==="
  ./predict.py "$img"
done
```

### 🎤 대사

> "Unit_test2는 Grape 4 클래스에서 뽑은 10장입니다. 같은 방식으로 예측합니다."

> "평가표가 명시적으로 'Unit_test2에서 다 틀리면 누수 의심하라'고 적혀 있습니다. 우리는 §7에서 설명드린 split→augment 순서를 지켜서 누수를 차단했고, val accuracy 99.79%가 자연스러운 결과라는 걸 Unit_test2도 통과하는 걸로 보여드릴 수 있습니다."

### ⚠️ 함정

- **10장 모두 틀리면 누수 의심 명시** → §7 누수 회피 설명을 한 번 더 강조할 준비.
- Unit_test2 이미지가 PlantVillage 외 source(예: 야외 휴대폰 사진)이면 자연스럽게 더 낮을 수 있음. confusion matrix로 "내부 데이터 분포에선 잘 동작한다"는 증거 제시.

---

## 10. 자주 받는 질문 (Q&A)

### Q1. "CNN을 왜 직접 만들었어요? Transfer learning이 더 쉽지 않아요?"

> "두 모델 모두 만들었습니다. 다만 default는 ScratchCNN으로 잡았습니다. 이유는 (1) 모델의 모든 레이어를 제가 직접 설계해서 한 줄씩 설명할 수 있다는 게 디펜스에 유리하고 (2) PDF가 '모델을 설명할 수 있어야 한다'고 명시했기 때문에 black-box 사전학습 모델보다 직접 만든 모델이 설명 면에서 안전합니다. Transfer는 비교용으로 두고 production에서 어떻게 쓰는지도 시연 가능합니다."

### Q2. "왜 EfficientNet-B0이에요? ResNet이나 다른 건?"

> "ImageNet에서 비슷한 정확도일 때 EfficientNet-B0이 파라미터/연산량 효율이 좋아서 modern 표준 baseline입니다. ResNet-50은 25M 파라미터인데 EfficientNet-B0은 5M 정도로 1/5 수준이라 CPU 추론에서도 빠릅니다."

### Q3. "데이터 누수가 정확히 뭐예요?"

> "누수는 모델이 평가 단계에서 보면 안 되는 정보를 학습 중에 이미 본 상태를 말합니다. 가장 흔한 형태가 augment-then-split입니다. 같은 원본의 변형(좌우 뒤집기 같은)이 train과 val에 흩어지면, 모델이 train에서 외운 답을 val에서 거의 그대로 만나서 가짜 100% 정확도가 나옵니다. 우리는 원본만 split하고 augmentation은 train batch에 메모리에서만 적용해서 차단했습니다."

### Q4. "Online augmentation이 정확히 뭐예요?"

> "디스크에 augmented 이미지를 안 만들고, 학습 중 batch를 만들 때마다 메모리에서 random 변형을 적용하는 방식입니다. `LeafDataset.__getitem__`에서 transform이 매 호출마다 새 random 값으로 실행됩니다. 같은 image (1)이 epoch마다 다른 변형으로 보이니까 사실상 무한 데이터 효과가 있고, 디스크에 안 남으니 val에 노출될 경로가 없습니다."

### Q5. "WeightedRandomSampler는 왜 필요해요? augmented_directory로 균형 맞추면 되잖아요?"

> "augmented_directory는 Part 2의 PDF 요구사항을 위해 만들지만 학습엔 안 씁니다(누수 때문). 그래서 클래스 불균형을 보정할 다른 방법이 필요하고, 그게 PyTorch의 `WeightedRandomSampler`입니다. 각 sample의 weight = 1/그 클래스 크기로 줘서 batch마다 작은 클래스가 자주 뽑히게 합니다. 디스크 저장 없이 같은 효과 + 누수 회피."

### Q6. "Stratified split이 일반 random split이랑 어떻게 달라요?"

> "일반 random split은 운나쁘게 val에 특정 클래스가 0장일 수도 있어서 정확도 측정이 불가능해질 수 있습니다. Stratified split은 클래스 비율을 유지하면서 8:2로 나눠서 모든 클래스가 val에 일정 비율 들어가게 보장합니다. `sklearn.train_test_split(stratify=labels)` 한 줄."

### Q7. "왜 seed=42인가요?"

> "Douglas Adams의 'Hitchhiker's Guide to the Galaxy'에서 '삶, 우주, 모든 것의 답'이 42라서 ML 커뮤니티에서 농담 같은 표준이 됐습니다. 숫자 자체는 의미 없고 단지 재현성(reproducibility)을 위해 고정값이면 됩니다. 같은 seed로 split하면 학습할 때마다 똑같은 train/val 분할이 나와서 결과 비교가 가능합니다."

### Q8. "정확도가 99%는 너무 좋은데, 진짜인가요?"

> "세 가지 증거를 제시할 수 있습니다. (1) PlantVillage는 통제된 회색 배경에서 잎이 정렬되어 있고 클래스 간 시각적 차이가 명확합니다. 학계 논문에서도 EfficientNet/ResNet으로 95-99%가 흔합니다. (2) ScratchCNN과 EfficientNet 두 다른 방식이 99.79%, 99.86%로 0.07pp 이내 → 데이터셋이 명확한 거지 한 모델의 트릭이 아닙니다. (3) confusion matrix가 자연스러워서 1-2장 misclassified가 클래스별로 흩어져 있습니다. 추가로 방금 `eval_val.py`로 즉시 재검증해 보였습니다."

### Q9. "Overfitting은 어떻게 막았나요?"

> "다섯 가지 도구를 동시에 씁니다. (1) Dropout(0.4) — 학습 중 40% 뉴런 끔, 특정 뉴런 의존 방지. (2) weight_decay=1e-4 — L2 regularization. (3) Online augmentation — train batch에 random 변형. (4) Early stopping(patience=5) — val 정체 시 중단. (5) ReduceLROnPlateau — val 정체 시 LR 절반. `learning_curves.png`에서 train과 val이 함께 수렴하는 게 증거입니다."

### Q10. "uv는 왜 쓰나요?"

> "pip + virtualenv + pyenv + pip-tools를 한 번에 대체하는 modern Python 도구입니다. Rust로 작성돼서 pip 대비 10-100배 빠르고 `uv.lock`으로 의존성 재현성을 보장합니다. 2024-2026 Python 생태계에서 표준으로 자리잡는 중입니다."

### Q11. "코드 어디 있어요?"

```
루트:           Distribution / Augmentation / Transformation / train / predict.py
실제 로직:      src/leaffliction/*.py
모델:           src/leaffliction/models/{scratch_cnn, transfer}.py
학습 루프:      src/leaffliction/trainer.py
추론:           src/leaffliction/predictor.py
변환:           src/leaffliction/transform.py
증강:           src/leaffliction/augment.py
시각화:         src/leaffliction/viz.py
테스트:         tests/test_*.py
스크립트:       scripts/eval_val.py / verify.sh / check_no_dataset.sh
```

### Q12. "테스트 있어요?"

> "네, `tests/` 디렉토리에 28개 pytest 케이스가 있습니다. `make test`로 실행. CI 없이도 로컬에서 sanity check 가능. `make smoke`는 entrypoint `--help`만 빠르게 확인."

### Q13. "Q&A 못 답한 게 있으면 어떻게 해요?"

> "솔직하게 '그건 잘 모르겠는데 코드 보면서 같이 보겠습니다'라고 하고 코드 띄움. 평가표 0점 조건은 '설명을 못 함'이지 '한두 가지 모름'이 아니라 정직성이 더 중요."

---

## 11. 위험 시나리오

### A. `make verify` 실패 (signature 불일치)

```bash
# 1. USB의 zip 무결성 재확인
ls -la trained_models.zip augmented_directory.zip
shasum trained_models.zip augmented_directory.zip

# 2. 백업 USB로 재시도

# 3. 그래도 안 되면 학습 재실행 (~50분)
./train.py images/ --epochs 25
```

### B. `uv sync` 실패 (plantcv 빌드)

```bash
uv pip install plantcv --no-build-isolation
# 또는 시스템 pip:
python -m pip install -e .
```

### C. matplotlib 창 안 뜨는 환경 (SSH/Docker)

```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png   # 또는 평가자 본인 노트북으로 scp
```

### D. "Unit_test에서 다 틀렸어요"

→ §7의 누수 회피 설명 한 번 더. `confusion_matrix.png`로 "내부 분포에선 잘 동작"을 보여주고 PlantVillage 외 데이터에선 자연스럽게 떨어질 수 있음을 인정.

### E. 명령이 안 됨 (`./Distribution.py: command not found`)

```bash
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py
# 또는 venv 활성화 확인:
source .venv/bin/activate
# 또는:
uv run python Distribution.py images/
```

### F. zip이 너무 커서 USB에 안 들어감

→ `augmented_directory.zip`(187MB)이 가장 큼. USB는 256MB 이상이면 충분. 클라우드 백업도 권장.

### G. 평가자가 다른 dataset path를 줌

```bash
./Distribution.py /path/to/their/Apple
./Augmentation.py "/path/to/their/Apple/apple_healthy/image (1).JPG"
```

모든 entrypoint가 path 인자를 받음. 우리 `images/`에 종속 안 됨.

---

## 12. 시간 배분 (30분 defense)

| 시간 | 단계 |
|------|------|
| 0-2 | git clone + uv sync + USB zip 복사 |
| 2-4 | Error Management (signature + flake8) |
| 4-7 | Part 1 — Distribution |
| 7-10 | Part 2 — Augmentation (6 sibling) |
| 10-12 | Part 1 verify — augmented_directory pie chart |
| 12-16 | Part 3 — Transformation (6 + histogram) |
| 16-18 | Part 4 (1) — eval_val.py 99.79% |
| 18-23 | **Part 4 (2) — 모델 설명 5점 (가장 큰 단일 점수)** |
| 23-26 | Part 4 (3) — Unit_test1 (Apple) |
| 26-29 | Part 4 (4) — Unit_test2 (Grape) |
| 29-30 | Q&A 여유 |

---

## 13. USB / 사전 점검 체크리스트

평가 전날:
- [ ] `make lint` 통과
- [ ] `make test` 통과
- [ ] `make verify` 통과
- [ ] `./scripts/eval_val.py images` ≥ 90%
- [ ] `./scripts/eval_val.py images --model transfer` ≥ 90%
- [ ] `git status`에서 `*.zip / *.pt / images/ / augmented_directory/` 없음

USB:
- [ ] `trained_models.zip` (~20MB)
- [ ] `augmented_directory.zip` (~187MB)
- [ ] 백업 USB 또는 cloud (Google Drive, AirDrop, scp)
- [ ] 노트북 충전기

평가 시작 5분 전:
- [ ] `cd ~/42/Leaffliction && source .venv/bin/activate`
- [ ] `artifacts/learning_curves.png` 미리 한 번 열어 확인
- [ ] `artifacts/confusion_matrix.png` 미리 한 번 열어 확인

---

## 14. 마무리 한 줄 pitch (Conclusion 칸용)

> "PDF의 5개 entrypoint와 평가표의 0점 함정(augmented_directory 균형, signature.txt 일치, 100+ 이미지 ≥90%)을 모두 통과 + 데이터 누수를 회피한 정직한 결과입니다. ScratchCNN을 처음부터 설계해 모든 레이어를 설명할 수 있고, EfficientNet-B0 transfer model도 비교용으로 두어 CNN의 production 활용 방식까지 함께 시연 가능합니다."

---

## 🔗 관련 문서

- 디자인 결정 + 근거: [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)
- 구현 plan: [docs/superpowers/plans/2026-04-28-leaffliction.md](superpowers/plans/2026-04-28-leaffliction.md)
- 산출물 무결성: `signature.txt` + `make verify`
