# Leaffliction — Defense Sheet

> Subject PDF + Evaluation PDF 기반. defense day에 그대로 따라가면 되는 시나리오 + 예상 질문 답변.

---

## 🗓️ D-1 자체 점검

```bash
cd ~/42/Leaffliction

# 1) 코드 품질
make lint                                # ruff + flake8 통과
make test                                # pytest 모두 통과
make smoke                               # 5 entrypoints --help OK

# 2) 산출물 무결성
make verify                              # signature.txt vs zip 해시 일치

# 3) 직접 시연 1회씩
./Distribution.py images/
./Augmentation.py "images/Apple_healthy/image (1).JPG"
rm "images/Apple_healthy/image (1)_"*.JPG     # 시연 후 정리
./Transformation.py "images/Apple_healthy/image (1).JPG"
./predict.py "images/Apple_healthy/image (1).JPG"

# 4) USB 백업
cp trained_models.zip augmented_directory.zip /Volumes/USB/

# 5) 시각화 미리 열어두기 (defense 시 빠른 응답용)
open artifacts/learning_curves.png
open artifacts/confusion_matrix.png
cat artifacts/classification_report.txt
cat artifacts/metadata.json
```

체크리스트:
- [ ] uv sync 모든 의존성 설치됨
- [ ] make verify 통과
- [ ] 5 entrypoint 모두 실행 OK
- [ ] USB에 zip 2개 복사
- [ ] git status 깨끗 (zip/pt 파일 git에 없음)
- [ ] 노트북 충전 + 어댑터

---

## 🎬 Defense Day — 단계별 시나리오

### 단계 0: 평가자 도착, 환경 setup (5분)

```bash
# 평가자가 본인 노트북에서 (또는 학생 노트북에서) 수행
git clone <repo_url> Leaffliction && cd Leaffliction
uv sync                              # 30초

# 학생: USB에서 zip 두 개 복사
cp /Volumes/USB/trained_models.zip .
cp /Volumes/USB/augmented_directory.zip .
```

### 단계 1: Error Management (eval PDF 필수 통과 항목) — 2분

**평가자가 가장 먼저 확인하는 것**:

```bash
# 1) signature 검증
make verify
# → OK trained_models.zip
# → OK augmented_directory.zip
# → All signatures verified.

# 또는 평가자가 직접:
shasum trained_models.zip augmented_directory.zip
cat signature.txt
# 두 출력이 일치해야 함 (diff로 비교 가능)

# 2) flake8 norm 확인
flake8 src tests *.py
# (출력 없음 = 통과)
```

**왜 중요?**: PDF Chapter V — signature 불일치 = **0점**. norm error = **0점**.

→ 우리는 두 가지 모두 통과 준비됨.

### 단계 2: Part 1 — Distribution.py (5분)

```bash
./Distribution.py images/
```

→ pie + bar 차트 표시 (subject PDF 예시와 동일 형식)

**보여줄 것**:
- 8개 클래스 분포
- 막대 차트에 정확한 숫자 라벨
- Apple_healthy: 1640 vs Apple_rust: 275 → **6배 불균형 발견**

**핵심 talking point**:
> "이 시각화로 데이터셋의 불균형을 즉시 인식할 수 있습니다. Apple_rust(275)와 Apple_healthy(1640)의 차이가 가장 큽니다. → 그래서 Part 2에서 균형화가 필요했습니다."

### 단계 3: Part 2 — Augmentation.py (단일 모드, 10분)

```bash
./Augmentation.py "images/Apple_healthy/image (1).JPG"
```

→ 화면에 7장 grid (Original + 6 변형). 같은 폴더에 sibling 파일 6개 생성.

```bash
# 파일 생성 확인
ls "images/Apple_healthy/image (1)"*
# image (1).JPG
# image (1)_Crop.JPG
# image (1)_Distortion.JPG
# image (1)_Flip.JPG
# image (1)_Rotate.JPG
# image (1)_Shear.JPG
# image (1)_Skew.JPG

# 정리
rm "images/Apple_healthy/image (1)_"*.JPG
```

**eval PDF 체크 6개 변형**: Flip / Rotate / Skew / Shear / Crop / Distortion — 모두 만족.

### 단계 4: Part 1 추가 검증 — augmented_directory 균형 확인 ⚠️ (eval PDF 핵심 함정)

```bash
./Distribution.py ./augmented_directory
```

→ 8개 조각 균등 (각 12.5%, 1640장씩).

**eval PDF 명시**:
> "If the pie chart is still the same as above you must put 0 to this exercise and exercise Part 1"

= augmented_directory가 균형 잡혀있지 않으면 **Part 1 + Part 2 둘 다 0점**.

→ 우리는 8 × 1640 = 13,120장 완벽 균형. 통과.

### 단계 5: Part 3 — Transformation.py (10분)

```bash
# 도움말 (PDF 명시: -h 지원)
./Transformation.py -h

# 단일 이미지 시연
./Transformation.py "images/Apple_healthy/image (1).JPG"
```

→ 한 figure에 6 plantCV 변환 + 9-channel color histogram (통합 표시).

다른 클래스로도 시연 추천:
```bash
./Transformation.py "images/Apple_Black_rot/image (1).JPG"
./Transformation.py "images/Grape_Esca/image (1).JPG"
```

(병변 있는 잎과 자연 구멍 있는 잎)

```bash
# 배치 모드도 짧게
mkdir -p /tmp/tx
./Transformation.py -src images/Apple_healthy/ -dst /tmp/tx -mask
ls /tmp/tx | head -3        # *_Mask.JPG 파일들
rm -rf /tmp/tx
```

### 단계 6: Part 4 — train.py 코드 + 결과 (15분)

**평가표**: "100+ 이미지에 90% 이상 정확도"

```bash
# 1) Validation 결과 (학습 시 이미 측정된 것)
cat artifacts/classification_report.txt
# accuracy 0.99xx > 0.90 ✅

cat artifacts/metadata.json
# ScratchCNN val_accuracy
# (--model both인 경우 transfer val_accuracy도)

# 2) 시각화
open artifacts/learning_curves.png        # train/val loss·acc 곡선
open artifacts/confusion_matrix.png       # 어떤 클래스가 헷갈리는지
```

### 단계 7: Part 4 모델 설명 (eval PDF: 0-5점, 가장 중요한 talking point)

**한 줄 요약**:
> "ScratchCNN을 처음부터 직접 설계했고, 비교용으로 EfficientNet-B0 transfer learning도 두었습니다. 둘 다 99%+ 정확도. 자세한 설명 드릴게요."

**핵심 4가지 설명** (각 1-2분):

1. **데이터 흐름**
   - 원본 7,221장 → stratified 80:20 split (클래스 비율 유지) → train 5,777 / val 1,444
   - `WeightedRandomSampler`로 불균형 보정 (Apple_rust 같은 minority도 batch마다 균등)
   - online augmentation: `RandomHorizontalFlip` + `RandomRotation` (메모리에서만)

2. **ScratchCNN 구조**
   - 4 conv blocks (3→32→64→128→256 채널)
   - 각 block = Conv-BN-ReLU-Conv-BN-ReLU-MaxPool
   - Layer 1-2: 선/모서리, Layer 3-4: 잎맥/잎종류
   - 마지막: AdaptiveAvgPool + Dropout(0.4) + Linear(256, 8)
   - 약 1.18M params (가벼움)

3. **학습 루프**
   - PyTorch 5줄 (forward → loss → zero_grad → backward → step)
   - CrossEntropyLoss + Adam(lr=1e-3, weight_decay=1e-4)
   - ReduceLROnPlateau 스케줄러 (val_acc plateau시 LR 절반)
   - Early stopping (patience=5)

4. **누수 방지**
   - 원본 `images/`로 학습 (augmented_directory 아님!)
   - 같은 원본의 augmentation이 train/val에 분리되는 사고 차단
   - val에는 augmentation 안 함 (정직한 측정)

### 단계 8: Unit_test1 (Apple 이미지) — 5분

```bash
# 평가자가 test_images.zip 다운로드
unzip ~/Downloads/test_images.zip -d /tmp/test_images
ls /tmp/test_images/Unit_test1
```

각 이미지에 대해:
```bash
./predict.py "/tmp/test_images/Unit_test1/Apple_healthy1.JPG"
# Class predicted: Apple_healthy (xxx%)
```

→ 파일 이름과 예측이 일치해야 1점. 10장 모두 맞추면 5/5.

### 단계 9: Unit_test2 (Grape 이미지) — 5분

```bash
./predict.py "/tmp/test_images/Unit_test2/Grape_spot.JPG"
# Class predicted: Grape_spot (xxx%)
```

**eval PDF 명시 경고**:
> "If the 10 images are misclassified, ask yourself how the student was able to get a good classification in his validation set."

= 10장 다 틀리면 데이터 누수 의심. → 우리는 v2 학습 (원본 images/)으로 누수 방지. 안전.

---

## ❓ 예상 질문 + 답변

### Q1. 데이터 누수 어떻게 방지했나요?

> 학습은 원본 `images/` 7,221장에서만 했습니다. `train_test_split(stratify=labels, random_state=42)`로 80:20 split하면 같은 이미지가 두 split에 동시에 들어가지 않습니다. 
> 
> Augmented_directory도 만들었지만 **거기엔 같은 원본의 변형들이 함께 있어서** split 시 train과 val에 그 변형들이 흩어질 수 있습니다 → 누수. 그래서 augmented_directory는 PDF Part 2 산출물로만 쓰고, 학습 입력은 깨끗한 원본만 사용. Augmentation은 학습 중 메모리에서만 (transforms.v2의 `RandomHorizontalFlip` + `RandomRotation`).

### Q2. 정확도 99%는 너무 좋은데?

> PlantVillage 데이터셋은 통제된 환경(균일 회색 배경)에서 촬영되어 클래스 간 시각적 차이가 명확합니다. 학계 논문에서도 EfficientNet/ResNet 계열이 95-99% 정확도가 흔합니다.
> 
> 추가 증거 3가지:
> 1. ScratchCNN(직접 설계)과 TransferModel(사전학습)의 결과 차이가 0.07pp 이내 → 데이터 자체가 명확 (특정 모델의 트릭이 아님)
> 2. learning_curves.png에 train과 val 곡선 갭이 작음 (overfitting 신호 X)
> 3. confusion_matrix.png에 시각적으로 비슷한 클래스 사이 1-2장 confusion은 존재 → 100%는 아님

### Q3. forward pass / backward pass 설명?

`trainer.py:_epoch` 보여주며:
```python
logits = model(x)              # forward
loss = criterion(logits, y)
optimizer.zero_grad()
loss.backward()                # backward (autograd가 chain rule 자동)
optimizer.step()
```
> PyTorch의 표준 5줄입니다. forward에서 모델이 예측, criterion이 정답과 비교해 loss 계산. `loss.backward()` 한 줄이 모든 weight에 대한 미분을 자동으로 chain rule로 계산하고, `optimizer.step()`이 미분의 반대 방향으로 weight를 살짝 업데이트합니다.

### Q4. ScratchCNN 구조 설명?

```python
self.features = nn.Sequential(
    _conv_block(3, 32),     # 색깔 → 작은 패턴
    _conv_block(32, 64),    # 작은 패턴 → 중간 패턴
    _conv_block(64, 128),   # 중간 → 큰 패턴
    _conv_block(128, 256),  # 큰 패턴 → 추상적 의미
)
```
> 각 conv_block은 두 번의 Conv-BN-ReLU + MaxPool. 3×3 Conv를 두 번 = 5×5 효과 + 비선형 두 번. MaxPool로 공간 절반, 채널 두 배 늘리며 점점 추상화. 마지막에 AdaptiveAvgPool로 공간 평균 후 Linear(256, 8)로 분류.

### Q5. Overfitting 어떻게 방지?

5가지:
1. **Dropout(0.4)** — 학습 중 40% 뉴런 무작위로 끔
2. **Weight decay** (1e-4) — L2 regularization
3. **Online augmentation** — train batch에 random 변형 (val은 깨끗)
4. **Early stopping** (patience=5) — val 정체 시 학습 멈춤
5. **ReduceLROnPlateau** — val 정체 시 LR 절반으로

learning_curves.png 보여주며:
> "이 곡선에서 train과 val이 함께 수렴하고, val이 train보다 살짝 위에 있는 epoch도 있습니다. 이는 정상 generalization 신호이고, overfitting의 정반대 패턴입니다."

### Q6. 왜 두 모델? (TransferModel은 왜 있나)

> ScratchCNN을 직접 설계해서 모든 라인을 본인이 설명할 수 있습니다. TransferModel은 EfficientNet-B0 — ImageNet으로 사전학습된 CNN — 도 같이 두어서 "직접 만든 CNN과 production에서 쓰는 CNN(transfer learning)을 비교"하는 talking point로 활용. EfficientNet-B0 자체가 CNN의 한 형태라, "CNN을 잘 이해했음"을 보여주는 추가 증거입니다. 기본은 ScratchCNN, 옵션으로 TransferModel.

### Q7. plantCV는 왜? OpenCV 안 쓰고?

> plantCV는 OpenCV 위에 식물 분석 특화 기능을 얹은 라이브러리입니다. PDF가 권장했고, 잎 이미지 분석에 최적화. 우리는 `_binary_mask`에서 **LAB chroma magnitude + Otsu threshold + connected components + binary fill holes**로 잎을 강건히 추출합니다. PlantVillage의 회색 배경에 잘 작동.

### Q8. Albumentations 6 변형 — Skew와 Shear 어떻게 다른가?

```python
"Skew":  A.Affine(shear={"x": (-15,15), "y": (-15,15)})   # 양방향
"Shear": A.Affine(shear={"x": (-25,25)})                   # 단방향
```
> 둘 다 `A.Affine`의 shear 파라미터를 다르게 설정. Shear는 x축 단방향 기울이기(평행사변형), Skew는 x+y 양방향(더 비뚤어진 모양). PDF가 둘을 별도로 요구했고, Affine으로 둘 다 표현 가능.

### Q9. signature.txt가 뭐고 왜 필요?

```bash
cat signature.txt
# <sha1>  trained_models.zip
# <sha1>  augmented_directory.zip
```
> 학습 산출물 zip의 SHA1 해시. 작은 텍스트라 git에 commit. defense 때 평가자가 `shasum *.zip` 결과와 비교 → 일치하면 "git push 후 학생이 모델을 바꿔치기 안 했음" 증명. PDF Chapter V 명시 요구사항.

### Q10. WeightedRandomSampler가 뭐?

> 데이터셋 불균형이 학습에 미치는 영향을 줄이는 PyTorch의 sampler. 각 sample의 weight를 그 클래스 크기의 역수로 줍니다. 즉 Apple_rust(220 train) 1장은 weight 1/220, Apple_healthy(1312 train) 1장은 weight 1/1312. → batch를 만들 때 작은 클래스가 자주 뽑힘. **batch마다 클래스가 균등**해지는 효과. 디스크에 augmentation을 더 저장하지 않아도 같은 효과.

---

## 🛡️ 위험 시나리오 + 대응

### A. `make verify` 실패
- USB의 zip이 손상됨 → 다른 USB로 재시도
- 새 zip 들고 와도 hash가 다르면 학습을 다시 돌리거나 평가 일정 다시 잡기

### B. `uv sync` 실패 (plantcv 빌드 오류)
```bash
uv pip install plantcv --no-build-isolation
```

### C. matplotlib 창 안 뜨는 환경
```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png
```

### D. 평가자가 "왜 정확도가 너무 좋아?" 의심
→ Q2 답변 + learning_curves.png + confusion_matrix.png로 자연스러운 학습 곡선 + 클래스 confusion 증거 보여주기.

### E. Unit_test 중 일부 틀림
→ "어떤 클래스가 어떤 클래스로 헷갈렸는지" confusion_matrix.png로 설명. PlantVillage 외 야외 사진은 자연스럽게 정확도 떨어질 수 있음.

### F. "코드 어디 있어요?"
```
루트:           Distribution.py, Augmentation.py, Transformation.py, train.py, predict.py
실제 로직:      src/leaffliction/*.py
모델:           src/leaffliction/models/{scratch_cnn,transfer}.py
학습:           src/leaffliction/trainer.py
추론:           src/leaffliction/predictor.py
테스트:         tests/test_*.py
스크립트:       scripts/verify.sh, scripts/check_no_dataset.sh
디자인 문서:    docs/superpowers/specs/, docs/superpowers/plans/
```

---

## 🎤 마무리 한 줄 pitch

> "PDF 요구사항을 모두 충족하면서 (5 entrypoints, flake8, signature.txt, augmented_directory 균형, 90%+ accuracy), 데이터 누수를 회피한 정직한 결과를 얻었습니다. ScratchCNN을 처음부터 설계했고, EfficientNet-B0 transfer learning도 비교용으로 두어서 학습 가치(scratch)와 production accuracy(transfer)의 trade-off를 정량적으로 보여줄 수 있습니다."

---

## 📋 시간 배정 (30분 defense 기준)

| 시간 | 활동 |
|------|------|
| 0-3 | 환경 setup + `make verify` |
| 3-5 | `make lint` + `make test` |
| 5-10 | Part 1 (Distribution) + augmented_directory 검증 |
| 10-15 | Part 2 (Augmentation 단일 모드) |
| 15-20 | Part 3 (Transformation) |
| 20-25 | Part 4 (학습 결과 + 모델 설명) |
| 25-30 | Unit_test1 + Unit_test2 + Q&A |

50-60분이면 더 여유 있게.

---

## 📦 USB 챙기기 체크리스트

- [ ] `trained_models.zip` (~5MB scratch만 또는 ~25MB scratch+transfer)
- [ ] `augmented_directory.zip` (~187MB)
- [ ] `signature.txt` (이건 git에 있지만 백업 가능)
- [ ] (옵션) defense_sheet.md 인쇄본

---

> 이 sheet는 [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)와 함께 사용. design 결정의 더 깊은 근거가 거기 있음.
