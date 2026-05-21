# Leaffliction — Defense Sheet

> Subject PDF + Evaluation PDF + 대화 내내 다룬 개념을 한 문서에 모은 defense day playbook.
>
> 구조: **(1)** D-1 자체 점검 → **(2)** 핵심 개념 deep dive (필수 암기) → **(3)** Defense day 단계별 시나리오 → **(4)** 예상 질문 Q&A → **(5)** 위험 시나리오 → **(6)** 시간 배정 / USB 체크리스트.

---

## 🗓️ (1) D-1 자체 점검

```bash
cd ~/42/Leaffliction
source .venv/bin/activate            # 또는 uv run 사용

# 1) 코드 품질 (필수, eval PDF Error Management 통과 조건)
make lint                            # ruff + flake8
make test                            # 28 pytest 모두 통과

# 2) 산출물 무결성
make verify                          # signature.txt vs zip 해시 일치

# 3) val set 재검증 (eval PDF "100+ 이미지 >=90%" 자동 통과)
./scripts/eval_val.py images
./scripts/eval_val.py images --model transfer

# 4) 5 entrypoint 1회씩 시연 연습
./Distribution.py images/
./Augmentation.py "images/Apple_healthy/image (1).JPG"
rm "images/Apple_healthy/image (1)_"*.JPG     # 시연 후 정리
./Transformation.py "images/Apple_healthy/image (1).JPG"
./Transformation.py -h
./predict.py "images/Apple_healthy/image (1).JPG"

# 5) 시각화 미리 열어두기
open artifacts/learning_curves.png artifacts/confusion_matrix.png
cat artifacts/classification_report.txt
cat artifacts/metadata.json
```

**USB 준비**:
- [ ] `trained_models.zip` (~20MB)
- [ ] `augmented_directory.zip` (~187MB)
- [ ] 백업 USB도 같이

---

## 🧠 (2) 핵심 개념 Deep Dive

defense에서 평가자가 물어볼 가능성이 높은 개념들. 각 항목은 **한 문장 정의 + 직관 비유 + 우리 코드 위치**.

### 2-1. CNN — Convolutional Neural Network

**정의**: 작은 필터(스탬프)로 이미지를 훑어 패턴을 찾는 신경망. 여러 layer를 거쳐 점점 추상적인 의미로 변환.

**직관**: 사람이 그림 볼 때:
- 처음엔 선·점 (저수준)
- 모서리·텍스처 (중간)
- 잎맥·반점 (고수준)
- 잎의 종류·질병 (가장 추상적)

CNN의 4 conv block이 이 4단계를 자동 학습.

**우리 코드**: `src/leaffliction/models/scratch_cnn.py`

```python
self.features = nn.Sequential(
    _conv_block(3, 32),     # RGB → 32 작은 패턴 채널
    _conv_block(32, 64),    # 32 → 64 중간 패턴
    _conv_block(64, 128),   # 64 → 128 더 큰 패턴
    _conv_block(128, 256),  # 128 → 256 추상적 패턴
)
```

### 2-2. 채널 (Channel)

**정의**: 채널 1개 = 256×256 흑백 지도 1장. 같은 사진의 한 가지 관점.

**직관**: 같은 도시의 여러 정보 지도:
- 채널 1: 도로
- 채널 2: 건물
- 채널 3: 공원
- …
- 채널 32: 어떤 패턴

처음 사진은 3 채널 (R, G, B). Conv 통과하면 32, 64, 128, 256 채널로 확장.

**픽셀 vs 채널**:
- 픽셀 = 위치 (256×256 = 65,536개)
- 채널 = 종류 (3개, 32개, … 256개)

### 2-3. Conv2d (합성곱)

**정의**: 작은 패턴 비교기(필터/스탬프)를 이미지의 모든 위치에 대보고 매칭 점수 계산. 결과는 "어디에 그 패턴이 있나" 지도.

**직관**: `nn.Conv2d(in_ch=3, out_ch=32, kernel_size=3)` = 3×3 크기의 스탬프 32개가 각자 사진을 훑음. 32개의 새 지도(채널) 생성.

**중요**: 스탬프 내용은 학습으로 자동 결정. 우리는 "가로선 검출"같이 정하지 않음. AI가 "잎 분류에 유용한 패턴 32가지"를 알아서 찾음.

### 2-4. BatchNorm, ReLU, MaxPool (Conv의 보조 부속)

**`BatchNorm2d`**: 각 layer 출력을 평균 0, 분산 1로 정규화 → 학습 안정.
- 비유: 자동 변속기. 입력 밝기·대비가 달라도 일관되게 처리.

**`ReLU`**: `f(x) = max(0, x)`. 음수→0, 양수→그대로.
- 왜 필요?: 신경망에 **비선형성** 부여. ReLU 없으면 layer 여러 개 쌓아도 결국 선형 모델(어떤 직선 하나).

**`MaxPool2d(2)`**: 2×2 영역에서 최대값만 남김 → 크기 절반.
- 비유: 줌아웃 버튼. 작은 디테일은 버리고 큰 그림에 집중.

**우리 `_conv_block` 한 블록 = `(Conv-BN-ReLU) × 2 + MaxPool`**
- 3×3 두 번 = 5×5 한 번 효과 + 비선형 두 번 → 표현력 ↑

### 2-5. Head — AdaptiveAvgPool, Dropout, Linear

```python
self.head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # (256, 16, 16) → (256, 1, 1)
    nn.Flatten(),             # → (256,)
    nn.Dropout(0.4),          # 학습 중 40% 끔
    nn.Linear(256, 8),        # 256 features → 8 class scores
)
```

**`AdaptiveAvgPool2d(1)` = GAP (Global Average Pooling)**:
- 마지막 conv 출력은 16×16 격자에 256개 패턴.
- 위치 정보 평균 → 256개 숫자 ("사진 전체에 256개 패턴이 얼마나 강하게 있나").
- 옛날 FC layer가 너무 무거웠던 문제 해결 + 위치에 더 robust.

**`Dropout(0.4)`**: 학습 중 40% 뉴런 무작위로 끔. "특정 뉴런 의존" 방지 → overfitting 완화.

**`Linear(256, 8)`**: 256 → 8 클래스 logit (점수). softmax 적용하면 확률.

### 2-6. PyTorch 학습 5줄

**정의**: 모든 PyTorch 학습의 99%를 차지하는 핵심 패턴.

```python
logits = model(x)              # ① forward — 예측
loss = criterion(logits, y)    # ② loss — 정답과 차이
optimizer.zero_grad()          # ③ 이전 gradient 지움
loss.backward()                # ④ backward — 미분 자동 계산
optimizer.step()               # ⑤ weight 살짝 업데이트
```

**우리 코드**: `src/leaffliction/trainer.py`의 `_epoch` 함수 64-67줄.

**비유**: 학생이 문제 하나 풀고 → 채점 → 분석 → 답안 수정 → 다음 문제. 수천 번 반복.

**Autograd의 마법**:
- `loss.backward()` 한 줄이 **수백만 개 weight의 미분을 chain rule로 자동 계산**.
- 손으로 작성하면 수천 줄 + 버그 천국.
- 우리는 `nn.Module`로 forward만 정의 → backward는 PyTorch가 자동.

### 2-7. Loss / Optimizer / Scheduler / Early stopping

**`nn.CrossEntropyLoss`**:
- 분류 문제의 표준 loss
- 정답 클래스의 확률이 1에 가까울수록 loss ≈ 0
- 다른 클래스일수록 loss ↑

**`optim.Adam(lr=1e-3, weight_decay=1e-4)`**:
- 각 weight마다 적정 step 크기 자동 조절 (적응적)
- SGD보다 빠른 수렴, modern 기본값
- `weight_decay` = L2 regularization (overfitting 완화)

**`ReduceLROnPlateau(factor=0.5, patience=2)`**:
- val_acc 2 epoch 정체 → learning rate 절반으로
- 학습 후반 미세 조정 도움

**Early stopping (`patience=5`)**:
- val_acc가 5 epoch 동안 개선 없으면 학습 멈춤
- overfitting 진입 직전 차단

→ 4가지 모두 overfitting 방지 + 학습 효율 향상의 표준 도구.

### 2-8. Train / Val 분리 (Stratified) + WeightedRandomSampler

**`train_test_split(stratify=labels, random_state=42)`**:
- 클래스 비율 유지하며 80:20 분할 (Apple_healthy 1640 → train 1312 / val 328)
- `random_state=42`로 재현성 보장 — 같은 split 항상 나옴

**왜 stratified?**: 일반 random split이면 운나쁘게 val에 Apple_rust가 0장일 수도 → 측정 불가.

**`WeightedRandomSampler`**:
- Train set 안에서도 클래스 불균형 보정 (Apple_rust 220, Apple_healthy 1312)
- 각 sample weight = `1 / 그 클래스 train 크기`
- batch마다 작은 클래스가 자주 뽑힘 → batch 내부 클래스 균등

```python
sample_weights = 1.0 / class_count[train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(...))
```

→ Apple_rust 220장이 epoch당 평균 ~5번 뽑힘 (replacement=True). Apple_healthy 같은 1312장은 1번 정도.

### 2-9. Online Augmentation (메모리에서 그때그때)

**정의**: 학습 중 batch 만들 때 메모리에서 random 변형 적용. 디스크에 저장 X. Val에는 적용 X.

**우리 코드**:

```python
# train.py — train_tf (train batch에만 적용)
train_tf = Compose([
    Resize((256, 256), antialias=True),
    RandomHorizontalFlip(p=0.5),     # 50% 확률로 뒤집기
    RandomRotation(degrees=15),       # -15° ~ +15° 랜덤
    Normalize(mean=ImageNetMean, std=ImageNetStd),
])

# val_tf (val batch — augmentation 없음)
val_tf = Compose([
    Resize((256, 256), antialias=True),
    Normalize(mean=ImageNetMean, std=ImageNetStd),
])
```

```python
# dataset.py — LeafDataset.__getitem__
def __getitem__(self, idx):
    img = Image.open(path).convert("RGB")
    tensor = F.to_image(img)
    tensor = F.to_dtype(tensor, dtype=torch.float32, scale=True)
    if self.transform is not None:
        tensor = self.transform(tensor)      # ← 매 호출마다 새 random 변형
    return tensor, label
```

**효과**: 같은 image (1)이 epoch마다 다른 random 변형으로 보임 → 사실상 무한 데이터.

### 2-10. 데이터 누수 — Augment 순서가 핵심

**핵심 원리**: `augment → split` (디스크 augmentation 후 random split) = 누수 위험.  
**올바른 순서**: `split → augment` (split 후 train에만 augment 적용) = 안전.

**Augmented_directory로 학습 시 누수 발생 메커니즘**:
```
augmented_directory/Apple_rust/
  ├─ image (1).JPG              ← 원본
  ├─ image (1)_Flip_0.JPG       ← image (1)의 좌우 뒤집기
  ├─ image (1)_Rotate_5.JPG     ← image (1)의 회전
  ↓
train_test_split (파일 단위 random)
  ├─ Train: image (1).JPG + image (1)_Rotate_5.JPG
  └─ Val:   image (1)_Flip_0.JPG       ← train의 image (1)과 거의 같은 이미지
       → AI가 외운 답을 그대로 인정 → val_acc 가짜 100%
```

**우리 회피**: 원본 `images/`로 학습 + online augmentation. 원본만 split하니 같은 image가 두 split에 동시 못 들어감 → 누수 0%.

**증거**: v1 (augmented로 학습) val_acc = 100% (의심), v2 (원본 + online) val_acc = 99.79% (자연스러움).

### 2-11. Transfer Learning — Two-stage Fine-tuning

**정의**: 다른 데이터셋(ImageNet)으로 학습된 모델을 가져와서 우리 문제만 추가 학습.

**왜 효과적?**: CNN의 초기/중간 layer는 **거의 모든 사진에 공통된 패턴**(선·곡선·텍스처)을 학습. 잎 사진에도 그대로 활용 가능. 마지막 분류층만 새로.

**우리 `TransferModel`** (`src/leaffliction/models/transfer.py`):
```python
backbone = efficientnet_b0(weights=DEFAULT)           # ImageNet 가중치 다운로드
backbone.classifier = nn.Sequential(                  # 마지막 층만 교체
    nn.Dropout(0.3),
    nn.Linear(1280, 8),                                # 1000 클래스 → 8 클래스
)
self.freeze()                                          # backbone 동결
```

**Two-stage** (`trainer.py`의 `unfreeze_after`):
- Stage 1 (epoch 1-5): backbone 동결, classifier만 학습
- Stage 2 (epoch 6+): 전체 unfreeze + LR 1/10로 fine-tune

**우리 결과**: epoch 5→6에서 val_loss 0.05 → 0.011 점프 = unfreeze 효과 명확.

### 2-12. plantCV Mask — LAB Chroma + Otsu + Fill Holes

**문제**: PlantVillage의 회색 배경에서 잎을 robust하게 분리해야 함. 단순 saturation threshold는 healthy 사과 잎(회색-녹색)에서 잘 안 됨.

**우리 해결책** (`src/leaffliction/transform.py:_binary_mask`):
```python
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
chroma = sqrt((a-128)² + (b-128)²)                    # 회색 중심에서의 거리
_, binary = cv2.threshold(chroma, 0, 255, OTSU)       # 자동 threshold
binary = morphologyEx(binary, MORPH_OPEN)             # 노이즈 제거
largest_cc = keep_only_largest_connected_component()  # 잎 본체만
filled = binary_fill_holes(largest_cc)                # 내부 구멍 메움
```

**왜 LAB chroma?**:
- 회색 배경: chroma ≈ 0 (a*≈128, b*≈128)
- 녹색 잎: chroma 큼 (b axis로 노란-녹색)
- 갈색 병변: chroma 큼 (a + b)

→ 색 종류 무관하게 "회색이 아닌 영역"이 강한 신호 → robust.

### 2-13. signature.txt — SHA1 위조 방지 인장

**정의**: 학습/데이터셋 zip의 SHA1 해시를 적은 파일. defense day 평가자가 zip과 비교 → 일치 안 하면 0점.

**PDF Chapter V 요구**:
> "the signature of the signature.txt file will be compared with the one of your data set. If the two of them are not identical, your grade will be 0."

**우리 `train.py`**가 자동 생성:
```
<sha1>  trained_models.zip
<sha1>  augmented_directory.zip
```

**검증**: `make verify` 또는 평가자가 직접 `shasum *.zip` 후 비교.

### 2-14. 5개 Entrypoint의 흐름

```
images/ (원본, 7,221장)
   │
   ├─► Distribution.py            → pie + bar chart (불균형 발견)
   │
   ├─► Augmentation.py (single)   → 6 변형 sibling 저장 (PDF 시연용)
   │
   ├─► Augmentation.py (batch)    → augmented_directory/ (8×1640=13,120) + .zip
   │
   ├─► Transformation.py          → plantCV 6 변환 + 9 채널 histogram (시각화)
   │
   └─► train.py (images/ 사용)
          ├─► artifacts/model_scratch.pt   (1.18M params)
          ├─► artifacts/model_transfer.pt  (4M params, opt-in)
          ├─► artifacts/metadata.json + curves + matrix + report
          ├─► trained_models.zip (위 산출물 압축)
          └─► signature.txt (두 zip 해시)
                │
                ▼
            predict.py — 새 이미지 분류
            eval_val.py — val 1,445장 재검증 (PDF 100+ 90%+ 요구)
```

---

## 🎬 (3) Defense Day 단계별 시나리오

### Step 0: 환경 setup (5분)

```bash
git clone https://github.com/keonwoo98/Leaffliction.git
cd Leaffliction
uv sync                              # 30초
source .venv/bin/activate

# USB에서 zip 복사
cp /Volumes/USB/trained_models.zip .
cp /Volumes/USB/augmented_directory.zip .
```

### Step 1: Error Management (2분, eval PDF 필수)

```bash
make verify                          # signature 검증
flake8 src tests *.py                # PDF norm 검사
```

→ 둘 다 통과해야 다음 단계. 우리는 통과 준비됨.

### Step 2: Part 1 — Distribution (3분)

```bash
./Distribution.py images/
```

평가표 체크: pie + bar chart 표시. 우리 추가 기능 = 막대 위에 숫자 라벨.

**talking point**:
> "8 클래스, 7,221장. Apple_rust(275) vs Apple_healthy(1640) = 6배 불균형. → Part 2의 동기."

### Step 3: Part 2 — Augmentation 단일 모드 (5분)

```bash
./Augmentation.py "images/Apple_healthy/image (1).JPG"
ls "images/Apple_healthy/image (1)"*   # 6 sibling 확인
rm "images/Apple_healthy/image (1)_"*.JPG  # 정리
```

→ PDF 명시 6 변형 (Flip, Rotate, Skew, Shear, Crop, Distortion) + 파일명 규칙.

### Step 4: Part 1 추가 검증 — augmented_directory 균형 ⚠️ (eval PDF 핵심 함정)

```bash
./Distribution.py ./augmented_directory
```

→ 8 조각 균등 (12.5% × 8 = 1640 × 8).

**eval PDF**: "분포가 원본과 동일하면 Part 1, Part 2 둘 다 0점". 우리는 통과.

### Step 5: Part 3 — Transformation (5분)

```bash
./Transformation.py -h                                      # PDF 명시 -h
./Transformation.py "images/Apple_healthy/image (1).JPG"
./Transformation.py "images/Apple_Black_rot/image (1).JPG"  # 갈색 병변 시연
```

→ 6 plantCV 변환 + 9 채널 color histogram 한 figure에. 평가자가 각 변환 설명 요구 가능 (→ §2-12 talking point).

### Step 6: Part 4 — 학습 결과 + 모델 설명 (10분, 가장 큰 점수)

```bash
cat artifacts/classification_report.txt    # accuracy 0.9986, support 1445
cat artifacts/metadata.json                # val_accuracy + best_epoch

open artifacts/learning_curves.png         # 학습 곡선
open artifacts/confusion_matrix.png        # 혼동 행렬
```

**모델 설명 흐름** (§2-1 ~ §2-9를 요약해서):
1. 데이터 흐름 (stratified split + WeightedRandomSampler + online aug)
2. ScratchCNN 구조 (4 conv block, Conv-BN-ReLU-MaxPool, GAP head)
3. PyTorch 5줄 학습 루프
4. 누수 방지 (원본 images/로 학습)

### Step 7: Part 4 정확도 재검증 (3분, PDF "100+ 90%+")

```bash
./scripts/eval_val.py images
# → 1442/1445 = 99.79% PASS

./scripts/eval_val.py images --model transfer
# → 1443/1445 = 99.86% PASS
```

→ val set을 재현(seed=42 stratified split) 후 model forward → 독립적 검증. train.py 출력을 안 믿어도 OK.

### Step 8: Unit_test1 (5분)

```bash
unzip ~/Downloads/test_images.zip -d /tmp/test_images
for img in /tmp/test_images/Unit_test1/*.JPG; do
  ./predict.py "$img"
done
```

→ 각 이미지의 예측이 파일명과 일치 = 1점.

### Step 9: Unit_test2 (5분)

```bash
for img in /tmp/test_images/Unit_test2/*.JPG; do
  ./predict.py "$img"
done
```

→ 같은 채점. eval PDF 경고: 10장 다 틀리면 누수 의심. 우리는 회피 (§2-10).

---

## ❓ (4) 예상 질문 + 답변 (확장)

### Q1. CNN은 어떻게 이미지를 "이해"하나?

> 4 단계로 점점 추상화합니다. 처음엔 작은 패턴 비교기 (3×3 필터)들이 사진을 훑어 "선·점이 어디 있나" 지도를 만듭니다. 그 결과를 다음 layer가 재료로 받아서 "모서리·텍스처"를, 그 다음은 "잎맥·반점"을, 마지막은 "잎의 종류·질병"이라는 고수준 의미를 추출합니다. 각 단계마다 필터 종류가 늘어나고(채널 ↑) 공간 해상도는 줄어듭니다(MaxPool로 ↓).

### Q2. 채널이 정확히 뭐?

> 한 채널 = 같은 사진을 한 가지 관점에서 본 흑백 지도. 처음엔 R/G/B 3개. Conv 통과하면 32, 64, 128, 256으로 늘어나는데 각 채널이 "다른 종류의 패턴이 어디 있나" 지도입니다. 도시 지도가 도로 지도, 건물 지도, 공원 지도로 나뉜 것과 비슷.

### Q3. 학습이 실제로 어떻게 일어나나? (forward / backward)

`trainer.py:_epoch` 화면에 띄우며:
```python
logits = model(x)              # ① forward
loss = criterion(logits, y)    # ② 정답과 차이
optimizer.zero_grad()
loss.backward()                # ③ 모든 weight의 미분 자동 계산
optimizer.step()               # ④ weight 살짝 업데이트
```
> PyTorch 5줄입니다. forward에서 모델이 예측, criterion이 loss 계산, `backward()`가 chain rule을 자동 적용해 수백만 weight의 미분을 한 번에 계산, `optimizer.step()`이 그 반대 방향으로 weight를 살짝 이동. 이걸 batch당 한 번씩, 25 epoch × 약 180 step = 4,500번 반복하면서 weight가 정답에 가까워집니다.

### Q4. 데이터 누수 어떻게 방지?

> 핵심은 **augment과 split의 순서**입니다. augmented_directory를 그대로 split하면 같은 원본의 변형(좌우 뒤집기 같은)이 train과 val에 흩어져서 AI가 외운 답을 그대로 인정받는 가짜 100% 정확도가 나옵니다. 우리는 원본 `images/`를 split한 후 train batch에만 메모리에서 random augmentation(transforms.v2.RandomFlip + RandomRotation)을 적용합니다. val에는 augmentation 없음. 변형이 디스크에 저장되지 않고 메모리에서만 잠깐 존재하니 val에 노출될 경로가 없음.

### Q5. 정확도 99%는 너무 좋은데?

> 세 가지 증거:
> 1. **PlantVillage 데이터셋 특성**: 통제된 회색 배경에 잎이 정렬되어 있고 클래스 간 시각적 차이가 명확. 학계 논문에서도 EfficientNet/ResNet으로 95-99%가 흔합니다.
> 2. **두 모델 비슷한 결과**: ScratchCNN(99.79%)과 TransferModel(99.86%)이 0.07pp 이내. 한 모델의 트릭이 아니라 데이터 자체가 명확하다는 증거.
> 3. **자연스러운 confusion matrix**: 8 클래스 중 4개는 100%, 4개는 1-2장 misclassified. PlantVillage 외 야외 사진은 자연스럽게 더 낮을 것.
>
> 추가로 `./scripts/eval_val.py images`로 즉시 재검증 가능합니다.

### Q6. Overfitting 어떻게 방지?

> 다섯 가지 도구를 동시 사용합니다:
> 1. **Dropout(0.4)** — 학습 중 40% 뉴런 끔, 특정 뉴런 의존 방지
> 2. **Weight decay (1e-4)** — L2 regularization
> 3. **Online augmentation** — train batch에 random 변형, val엔 없음
> 4. **Early stopping (patience=5)** — val 정체 시 학습 중단
> 5. **ReduceLROnPlateau** — val 정체 시 LR 절반
>
> `learning_curves.png`에서 train과 val 곡선이 함께 수렴하고, train_loss > val_loss인 epoch도 있는 것을 보여드릴 수 있습니다. 이는 overfitting의 정반대 패턴 (online augmentation이 train을 어렵게 만들어서).

### Q7. ScratchCNN의 각 부속을 설명해보세요

> `_conv_block(in_ch, out_ch)`는 다음 7 단계:
> 1. **Conv2d(3×3, padding=1)**: 작은 패턴 비교기. `out_ch`개의 새 채널 생성.
> 2. **BatchNorm2d**: 출력을 평균 0, 분산 1로 정규화. 학습 안정화.
> 3. **ReLU**: 음수→0, 양수→그대로. 비선형성 부여 (없으면 단순 선형 모델).
> 4. 같은 conv 한 번 더 (5×5 효과 + 비선형 두 번).
> 5. **MaxPool2d(2)**: 2×2 영역 최대값. 크기 절반, 핵심만 유지.
>
> 4 block을 거치며 (3, 256, 256) → (256, 16, 16)로 변환. Head는 `AdaptiveAvgPool2d(1)`로 공간 평균 → 256개 숫자 → `Dropout(0.4)` → `Linear(256, 8)`로 8 클래스 점수.

### Q8. Transfer Learning이 왜 효과적?

> EfficientNet-B0는 ImageNet 100만 장(개, 고양이, 차, 꽃 등 1000 클래스)으로 학습됐습니다. CNN의 초기·중간 layer가 학습한 능력 — "선·곡선·텍스처·모서리·작은 모양 인식" — 은 잎 사진에도 그대로 활용 가능합니다. 우리는 마지막 1000-class 분류층만 8-class로 갈아끼우고, 잎 사진으로 추가 학습.
>
> Two-stage 학습: Stage 1(epoch 1-5)은 backbone 동결 + classifier만 학습, Stage 2(epoch 6+)는 전체 unfreeze + LR 1/10로 미세조정. 우리 결과에서 epoch 5→6 사이 val_loss 0.05→0.011 점프가 unfreeze 효과를 보여줍니다.

### Q9. WeightedRandomSampler가 뭐?

> 클래스 불균형을 batch 단위에서 보정하는 PyTorch sampler. 각 sample의 weight = `1 / 그 클래스 크기`. 예: Apple_rust(220장)의 한 sample = weight 1/220, Apple_healthy(1312장)의 한 sample = weight 1/1312. → batch 만들 때 작은 클래스가 자주 뽑힘. 결과적으로 batch마다 클래스가 거의 균등.
>
> Augmented_directory에 변형을 디스크에 저장하지 않아도 같은 효과 + 누수 방지.

### Q10. plantCV mask가 어떻게 동작?

> `_binary_mask` 함수:
> 1. RGB → LAB 색공간 변환
> 2. **Chroma magnitude** = sqrt((a-128)² + (b-128)²) 계산. 회색 배경은 ≈ 0, 잎(녹색)과 병변(갈색)은 큰 값.
> 3. **Otsu threshold**로 자동 cutoff 결정.
> 4. Morphological opening으로 작은 노이즈 제거.
> 5. **Largest connected component만 유지** (잎 본체).
> 6. **binary_fill_holes**로 잎 내부 구멍 메움.
>
> PlantVillage의 회색 배경 + 다양한 잎 색깔에 robust. 단순 HSV-S threshold는 healthy 사과 잎에 안 됐는데 chroma는 잘 됨.

### Q11. signature.txt가 뭐?

> 학습/데이터셋 zip의 SHA1 해시 목록. PDF Chapter V는 dataset을 git에 올리면 0점이라 명시 + zip의 hash가 signature.txt와 다르면 0점. 우리 `train.py`가 자동 생성:
> ```
> <sha1>  trained_models.zip
> <sha1>  augmented_directory.zip
> ```
> `make verify` 또는 `shasum *.zip` 직접 실행 후 `signature.txt`와 비교 가능.

### Q12. uv를 쓰는 이유?

> pip + virtualenv + pyenv + pip-tools를 통합한 modern Python 도구 (Rust 작성). pip 대비 10-100배 빠름 + lockfile(`uv.lock`)로 재현성 보장 + Python 버전 관리 통합. 2024-2026 Python community 표준으로 자리잡는 중.

### Q13. 왜 두 모델? 하나로 충분 아닌가?

> PDF가 모델 수를 강제하지 않습니다. ScratchCNN 하나로도 99.79% PDF 통과. 다만 두 모델을 두면 두 가지 이점:
> 1. **검증 의심 회피**: 두 다른 방식(직접 설계 vs 사전학습)에서 비슷한 결과 → 데이터셋이 명확한 거지 한 모델의 트릭 아님
> 2. **CNN 이해 시연**: EfficientNet-B0 자체가 CNN의 한 형태 → "CNN을 직접 만들 줄도 알고, production에서 transfer learning으로 쓰는 방식도 안다" 시연
>
> Default는 ScratchCNN(headline), Transfer는 `--model transfer/both` opt-in.

---

## 🛡️ (5) 위험 시나리오 + 대응

### A. `make verify` 실패
- USB의 zip이 손상 → 백업 USB로 재시도
- hash가 진짜 다르면 학습 다시 (~50분) 또는 평가 재일정

### B. `uv sync` 실패 (plantcv 빌드 오류)
```bash
uv pip install plantcv --no-build-isolation
```

### C. matplotlib 창 안 뜨는 환경 (SSH, Docker)
```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png
```

### D. "정확도 너무 좋아" 의심
→ §2-10, Q5 답변 + learning_curves.png + confusion_matrix.png + `./scripts/eval_val.py`로 즉시 재검증.

### E. Unit_test에서 일부 틀림
→ confusion_matrix.png로 "어떤 클래스가 어떤 클래스로 헷갈렸는지" 설명. PlantVillage 외 이미지는 자연스럽게 정확도 낮을 수 있음. 모델은 PlantVillage style에 최적화됨.

### F. "코드 어디 있어요?"
```
루트:           Distribution / Augmentation / Transformation / train / predict.py
실제 로직:      src/leaffliction/*.py
모델:           src/leaffliction/models/{scratch_cnn, transfer}.py
학습 루프:      src/leaffliction/trainer.py
추론:           src/leaffliction/predictor.py
테스트:         tests/test_*.py
스크립트:       scripts/verify.sh, scripts/eval_val.py, scripts/check_no_dataset.sh
디자인 문서:    docs/superpowers/specs/, plans/
```

### G. "venv 비활성화 상태에서 명령 안 됨"
```bash
source .venv/bin/activate
# 또는
uv run python Distribution.py ...
```

---

## 🎤 (6) 마무리 한 줄 pitch

> "PDF 요구사항을 모두 충족(5 entrypoints, flake8, signature.txt, augmented_directory 균형, 100+ val 90%+) + 데이터 누수를 회피한 정직한 결과. ScratchCNN을 처음부터 설계해 모든 라인을 설명할 수 있고, EfficientNet-B0 transfer model도 비교용으로 두어 CNN의 production 사용 방식까지 시연 가능합니다."

---

## 📋 (7) 시간 배정

**30분 defense**:
| 시간 | 활동 |
|------|------|
| 0-3 | env setup + make verify |
| 3-5 | make lint + make test |
| 5-8 | Part 1 (Distribution + augmented_directory 균형) |
| 8-13 | Part 2 (Augmentation 단일 모드) |
| 13-18 | Part 3 (Transformation) |
| 18-23 | Part 4 (학습 결과 + 모델 설명) |
| 23-26 | eval_val.py + Unit_test1 |
| 26-30 | Unit_test2 + Q&A |

**60분 defense**: 위 시간 두 배 + 개념 deep dive 더 자세히.

---

## 📦 (8) USB 체크리스트

- [ ] `trained_models.zip` (~20MB)
- [ ] `augmented_directory.zip` (~187MB)
- [ ] 평가 컴퓨터에 미리 옮겨둠 — verify로 무결성 확인 가능
- [ ] 백업 USB 또는 외부 채널 (AirDrop, scp)
- [ ] 노트북 충전 + 어댑터

---

## 🔗 관련 문서

- 디자인 결정 + 근거: [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)
- 구현 plan: [docs/superpowers/plans/2026-04-28-leaffliction.md](superpowers/plans/2026-04-28-leaffliction.md)
- 코드 / 산출물 무결성: `signature.txt` + `make verify`
