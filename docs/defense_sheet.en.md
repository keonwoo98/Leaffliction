# Leaffliction — Defense Script (English)

A walkthrough script you can read aloud while running the demo. Each section has (1) what the evaluator checks, (2) the command, (3) what to say, (4) related code, (5) common pitfalls.

---

## 0. Project Intro

🎤 **When you start the defense:**

> "Leaffliction is a computer vision project that classifies leaf photos — figuring out whether the leaf has a disease, and which one, or if it's healthy. We have 8 classes total: 4 Apple states (healthy, Black_rot, rust, scab) and 4 Grape states (healthy, Black_rot, Esca, spot). The dataset came from the subject and has about 7,200 images. They're controlled photos — gray background, one leaf per image, taken in a lab-style setup."
>
> "The project has 4 Parts. Part 1 is EDA, exploring the dataset. Part 2 is data augmentation. Part 3 is feature visualization using plantCV. Part 4 is where we actually train a CNN and run predictions. So we cover the whole ML pipeline — looking at data, preprocessing, training, and inference."

🎯 Four key design decisions (in case they ask):
- Main model is a ScratchCNN we designed from scratch. EfficientNet-B0 transfer learning is included as a comparison.
- We train on the original `images/` only. Augmentation is applied in memory at batch time, never written to disk. This avoids data leakage.
- For class imbalance we use `WeightedRandomSampler` at the batch level — no need to duplicate images on disk.
- All random operations are seeded with `seed=42` for reproducibility.

---

## 1. Error Management — Signature & Norm Check

### 🎯 What the evaluator checks
The hash in `signature.txt` must match the actual SHA1 of the `.zip` files. They use `diff` for that. For Python they also run `flake8` for norm. **If signatures don't match, the evaluation stops immediately.**

### 💻 Commands
```bash
make verify                  # auto-diff between signature.txt and actual hashes
flake8 src tests *.py        # norm check
```

### 🎤 What to say

> "So first I'll verify the signature. The `signature.txt` file records the SHA1 hashes of the two zips — `trained_models.zip` and `augmented_directory.zip` — generated right after training. This step checks if anyone tampered with the files when I moved them via USB. PDF Chapter V explicitly lists this as a zero-point condition."
>
> "SHA1 is a cryptographic hash, so even if a single byte of the zip changes, the hash is completely different. That's why this gives reliable integrity verification. `make verify` just ran that — clean output, signatures match."
>
> "Next is `flake8` for code style. 42 enforces norm via the evaluation rubric, and a single PEP8 violation means zero points. I have ruff and flake8 both hooked into pre-commit, so it's been verified."

### 📂 Related code
- `src/leaffliction/signature.py` — computes SHA1 and writes/verifies `signature.txt`. Uses only `hashlib`.
- `scripts/verify.sh` — what `make verify` calls.
- `Makefile` — shortcuts for `lint`, `verify`, `test`, `format`.

---

## 2. Part 1 — Distribution (Dataset Analysis)

### 🎯 What the evaluator checks
Running `./Distribution.py ./images` should display a pie chart.

### 💻 Command
```bash
./Distribution.py ./images
```

### 🎤 What to say

> "Part 1 is EDA — exploratory data analysis. Basically, before you put data into a model, you should look at it first. This is the first step in any ML project, because you need to understand the distribution before you can design augmentation strategies or pick a model."
>
> "`Distribution.py` reads each subfolder of `images/` as a class, counts the images, and shows both a pie chart and a bar chart. The pie shows ratios at a glance, the bar gives you the exact counts. I added number labels on top of each bar too."

When the chart shows up:

> "So you can see — 8 classes, about 7,228 images total. `Apple_healthy` has 1,640 which is the biggest, and `Apple_rust` has only 275, the smallest. That's roughly a **6× imbalance**. This is the class imbalance problem, and it's why Part 2 data augmentation exists. If you trained directly on this imbalance, the model would just learn to predict the majority class."

### 📂 Related code
- `src/leaffliction/dataset.py::discover_classes` — uses `pathlib.rglob("*.JPG")` to find all images and group them by parent folder name (= class). Returns a dict. This same function is reused in Part 2 and Part 4 so class labels stay consistent everywhere.
- `src/leaffliction/viz.py::plot_distribution` — uses matplotlib + seaborn to put pie + bar in one figure. Numbers on top of bars via `ax.annotate`.

---

## 3. Part 3 — Augmentation (6 Variants of One Image)

### 🎯 What the evaluator checks
Feeding one image should produce 6 sibling files. Filenames must be `image (1)_Flip.JPG`, `_Rotate.JPG`, `_Skew.JPG`, `_Shear.JPG`, `_Crop.JPG`, `_Distortion.JPG`.

### 💻 Commands
```bash
./Augmentation.py "images/Apple_healthy/image (1).JPG"
ls "images/Apple_healthy/image (1)"*
```

### 🎤 What to say

> "Part 2 is data augmentation. The idea is that if you flip a leaf photo horizontally, it's still the same disease. Same with rotation. So if you add those variants to your training data with the same label, you can artificially expand a small dataset and make the model more robust."
>
> "The PDF specifies 6 transformations — Flip, Rotate, Skew, Shear, Crop, and Distortion — applied to the same image. I implemented this with the Albumentations library."

When `ls` output is shown:

> "You can see 6 sibling files next to the original — `image (1)_Flip.JPG` and so on. The filenames exactly match the PDF specification."

If they ask about what each transform does:

> "Flip is horizontal mirror. Rotate is ±30 degrees, and the empty space gets filled with black. Skew is shear on both x and y axes — it warps the image like a parallelogram. Shear is x-axis only but with a stronger range, ±25 degrees. Crop takes a random 70-100% region and resizes it back to 256x256, so the leaf appears slightly larger. The last one, Distortion, simulates camera lens distortion — visually it's subtle on leaves since there are no straight lines, but pixel coordinates shift slightly."
>
> "Some of these may look similar to the human eye, but the key is that even if humans can't tell the difference, the model sees different pixel values every time. That helps it learn more general patterns instead of memorizing specific images."

When done:
```bash
rm "images/Apple_healthy/image (1)_"*.JPG
```

> "Let me clean up the 6 siblings so they don't affect the next step."

### 📂 Related code
- `src/leaffliction/augment.py`
  - `AUGMENTATION_OPS` — a dict defining the 6 ops. Keys are the suffix names (`Flip`, `Rotate`, etc.), values are Albumentations transform objects.
  - `apply_op(name, image)` — runs a single op on a numpy array, returns the transformed array.
  - `balance_directory(...)` — the batch mode core. Copies originals + augments to fill up + auto-zips.

### ⚠️ Pitfalls
Filenames must exactly match the PDF — `_Flip.JPG`, `_Rotate.JPG`, etc. We match.

---

## 4. Part 1 Additional Check — augmented_directory Balance ⚠️ Zero-Point Trap

### 🎯 What the evaluator checks
Running the same `Distribution.py` on `augmented_directory`. **All 8 pie slices must be equal.** If not, both Part 1 AND Part 2 get zero.

### 💻 Commands
```bash
# if the zip isn't unpacked yet
unzip -q augmented_directory.zip

./Distribution.py ./augmented_directory
```

### 🎤 What to say

> "Now I'll run the same `Distribution.py` on `augmented_directory`. This is the part the rubric explicitly warns about — if the pie chart still looks imbalanced like the original, both Part 1 and Part 2 get zero."

When the chart appears:

> "You can see 8 slices, exactly equal. 8 × 1,640 = 13,120 images. We picked the largest class, `Apple_healthy` at 1,640, as the target and filled up the other 7 classes with augmentation until each reached 1,640. Each slice is exactly 12.5%."
>
> "One thing to mention — this `augmented_directory` exists because of the PDF requirement, but we don't actually use it for training. Training on it would cause data leakage. I'll explain that in detail when we get to the model in Part 4."

### 📂 Related code
- `src/leaffliction/augment.py::balance_directory` — the algorithm:
  1. `discover_classes` to get images per class.
  2. `target = max(...)` — pick the largest class size.
  3. For each class: copy originals → augment the shortfall with random ops from the 6.
  4. Auto-call `zip_directory()` at the end to produce `augmented_directory.zip` for the signature.

---

## 5. Part 3 — Transformation (Visualizing Leaf Features with plantCV)

### 🎯 What the evaluator checks
Feeding one image should display 6 transformations + a 9-channel color histogram. They may ask you to explain each one.

### 💻 Commands
```bash
./Transformation.py "images/Apple_healthy/image (1).JPG"

# option check (PDF requires -h)
./Transformation.py -h

# batch mode
./Transformation.py -src images/Apple_healthy -dst /tmp/out -mask
```

### 🎤 What to say

#### Why we do Part 3 first

> "Part 3 visualizes leaf features using plantCV. Honestly, this is **how computer vision was done before deep learning**. Before CNNs, people would manually extract features — leaf area, color distribution, shape keypoints — and feed them to traditional classifiers like SVM or Random Forest. The innovation of CNN is automating that feature extraction step."
>
> "So Part 3 has two secondary purposes too. First, it gives us hands-on experience with plantCV, a domain-specific library that biologists and agricultural researchers actually use. Second, it serves as a **data sanity check** — we can visually verify that the mask works, and that color distributions actually differ between classes."

#### The 6 panels (point to each)

> "Original is just the raw RGB."
>
> "Gaussian blur smooths small noise with a 3×3 Gaussian filter. It's preprocessing — makes the next step's mask cleaner."
>
> "Mask is the most important of these. It separates the leaf from the background as a binary mask. If this fails, everything downstream breaks."
>
> "ROI is the mask with a bounding box overlay. Analyze object computes shape metrics — area, perimeter, centroid, aspect ratio. The last one, Pseudolandmarks, places keypoints along the leaf edge — it's a standard tool for shape comparison."

#### If they ask about the mask in detail

> "The mask was a bit tricky. My first attempt used HSV saturation thresholding, but healthy Apple leaves are gray-green so the saturation is too low — it only caught half the leaf. So I switched to **LAB color space chroma magnitude**. In LAB, the a-axis is green-red and b-axis is blue-yellow. Gray background has both a and b near 128, so chroma — `sqrt((a-128)² + (b-128)²)` — is close to 0. Green leaves and brown lesions both have large chroma. So I compute chroma magnitude, apply Otsu auto-thresholding, then morphological opening to remove noise, keep only the largest connected component, and finally fill internal holes with scipy's `binary_fill_holes`."

#### 9-channel color histogram

> "Below the 6 panels is the 9-channel color histogram. The X-axis is pixel intensity from 0 to 255, the Y-axis is how many pixels in the image have that value. Each curve is one channel's distribution."
>
> "Why 9 channels? Because **we look at 3 color spaces at once**: RGB 3 + HSV 3 + LAB 3."
>
> "**RGB** is the computer's native color space — R, G, B for red, green, blue intensity. The downside is that color and brightness are mixed together. If lighting changes, R, G, B all shift together so you can't easily isolate color changes."
>
> "**HSV** separates color intuitively. H (Hue) is what color it is — 0 is red, 120 is green, 240 is blue. S (Saturation) is how vivid the color is — 0 is gray, 255 is fully saturated. Gray backgrounds have low S, green leaves have high S. V is brightness."
>
> "**LAB** is perceptually uniform — designed to match human vision. L is lightness, a* is the green-red axis (negative = green), b* is blue-yellow (positive = yellow). **Green leaves push a* into negative, brown lesions push b* into positive**. That's why we used LAB for the mask too."
>
> "Why use all three? Because no single color space captures everything. RGB is native but lighting-sensitive. HSV separates color but isn't perceptually uniform. LAB matches human perception but isn't a native representation. Together they describe the same image from different angles, and that's a richer representation."
>
> "**Each class has a visually distinct distribution.** Healthy leaves show a tall G peak and a* pushed negative. Black_rot has b* extended further positive because of the brown lesions, plus a dark tail on L. Rust creates a second peak in H around 30-60 because of the orange spots."
>
> "In the old days, people would take these histograms directly as feature vectors — say 9 channels × 32 bins = 288 dimensions — and feed them to SVM or Random Forest. Accuracy was around 70-85%. CNNs automatically learn statistics like this inside their conv blocks and pushed accuracy up to 95-99%."

### 📂 Related code
- `src/leaffliction/transform.py`
  - `_binary_mask(rgb)` — the LAB chroma → Otsu → opening → largest CC → fill_holes pipeline. The mask core.
  - `gaussian_blur`, `mask`, `roi`, `analyze_object`, `pseudolandmarks` — the 6 transformation functions.
  - `color_histogram(rgb)` — computes the 9-channel histogram.
- `src/leaffliction/viz.py::plot_transformations` — uses `subplot2grid((2,6))` to put 6 transforms on top and a 6-column-spanning histogram below.

---

## 6. Part 4 (1/4) — Classification Accuracy (≥ 90%)

### 🎯 What the evaluator checks
Run on at least 100 test images and get ≥ 90% accuracy.

### 💻 Commands
```bash
./scripts/eval_val.py ./images
./scripts/eval_val.py ./images --model transfer
```

### 🎤 What to say

> "The first item in Part 4 is the 90% accuracy requirement on 100+ images. I made a script called `eval_val.py`. Let me explain how it works."
>
> "When `train.py` was learning, it did a stratified 80/20 split with `random_state=42`. So 1,445 out of the 7,228 images were held out as val, and the model **never saw any of those 1,445 during training**. `eval_val.py` uses the same `seed=42` to reproduce the split and pull out exactly the same 1,445 images, then runs forward pass only and computes accuracy. That's 14× more samples than the PDF's 100 minimum."

When the result appears:

> "1,442 out of 1,445, **99.79%**. That's about 10 percentage points above the 90% requirement. The per-class breakdown is also shown — all 8 classes above 99%."

For transfer:

> "The transfer model gets 99.86%, slightly higher. The gap is only 0.07 percentage points, which means the dataset is clear enough that both models can solve it — it's not a trick of one particular model."

### 📌 Heads up — images the val set got wrong (so you can answer if asked)

**ScratchCNN (3 wrong)**:
- `Grape_spot/image (128).JPG` → Apple_scab (57.1%)
- `Apple_healthy/image (1040).JPG` → Apple_Black_rot (39.7%)
- `Apple_healthy/image (1326).JPG` → Grape_healthy (44.1%)

**TransferModel (2 wrong)**:
- `Apple_healthy/image (1040).JPG` → Apple_scab (99.1%)
- `Grape_Black_rot/image (56).JPG` → Grape_Esca (93.7%)

> "Interesting thing — `Apple_healthy/image (1040).JPG` is **wrong in both models**. It's probably a genuinely difficult image, maybe a labeling noise. Also, ScratchCNN's wrong predictions have low confidence — 39 to 57% — which means the model honestly hesitated. TransferModel's wrong predictions have 93 to 99% confidence — it's confidently wrong. That's a typical calibration difference in neural networks."

### 📂 Related code
- `scripts/eval_val.py` — loads `LeafDataset` → reproduces val indices via `train_test_split(stratify=labels, random_state=42, test_size=0.2)` → loads weights from `trained_models.zip` → forward pass → per-class accuracy.

---

## 7. Part 4 (2/4) — Model Explanation ⭐ (5 points, most important section)

Order of explanation: (1) big picture → (2) data flow → (3) model architecture → (4) training loop → (5) leakage avoidance.

### 7-1. Why CNN?

🎤

> "It's an image classification problem, so I went with a CNN — Convolutional Neural Network. A CNN scans the image with small filters to find patterns, and then layers those into progressively more abstract meaning through multiple stages."
>
> "Intuitively it's like how humans look at a picture. First you see low-level features like edges and dots, then mid-level like corners and textures, then higher-level like leaf veins and spots, then finally 'this is an apple leaf with rust.' Our 4 conv blocks learn this 4-stage abstraction automatically."
>
> "The key thing is **the human doesn't tell the model 'these features matter.'** The model discovers from the data what's useful for classification. That's the essence of deep learning."

### 7-2. How the data flows (before training starts)

#### Stratified 80/20 split + seed=42

🎤

> "Data goes through 3 preprocessing steps before training."
>
> "First is train/val split. I call `sklearn.train_test_split` with `stratify=labels` and `random_state=42`. The 'stratified' part is important — a normal random split would shuffle all 7,228 images together and take 20%, but you might unluckily end up with zero `Apple_rust` images in val. Then you can't measure accuracy for that class at all. **Stratified extracts 20% from each class independently**, so every class is split 80/20 exactly."
>
> "The result is 5,783 train and 1,445 val, with val getting exactly the right ratios per class — Apple_Black_rot 124, Apple_healthy 328, Apple_rust 55, and so on."
>
> "`seed=42` is for reproducibility. The thing is, computer random is actually a deterministic function — same seed gives the same sequence. So when train.py splits with seed=42 and eval_val.py also splits with seed=42, **exactly the same 1,445 images come out as val**. That's how eval_val.py can verify against unseen data. 42 itself is a meaningless number — it's just a common convention in the ML community, I used it because it's a familiar default."

#### WeightedRandomSampler

🎤

> "Second step is class balance. Even after the 80/20 split, the class imbalance remains. Apple_rust has 220 in train, Apple_healthy has 1,312. If you train directly, batches will be flooded with Apple_healthy and the model will bias toward the majority class."
>
> "So I use `WeightedRandomSampler`. How it works — each sample gets a weight, and **weight = 1 / (size of the class that sample belongs to)**. One Apple_rust sample has weight 1/220, one Apple_healthy sample has 1/1312. Small-class samples have 6× larger weight."
>
> "Mathematically, the sum of weights within a class is `class_size × (1/class_size) = 1`. All classes sum to 1, so the total weight is 8 — same for every class. That means when you look at it class by class, **all 8 classes have equal sampling probability**, which gives you balanced batches."
>
> "In practice, in one epoch, each Apple_rust image gets picked an average of 3.3 times, while each Apple_healthy image only 0.55 times. Small classes are repeated, large classes are subsampled — and crucially **we don't duplicate any image on disk**, so there's no data leakage risk."

#### Online augmentation — different from Part 2

🎤

> "The third step is augmentation, and **this is where it gets different from Part 2**. People confuse this, so I want to be clear about it."
>
> "Part 2's augmentation writes augmented files to disk and creates `augmented_directory`. It uses all 6 ops (Flip, Rotate, Skew, Shear, Crop, Distortion) and is done once. **You should not train on this directly** — that's the data leakage trap. So `augmented_directory` is only used for the PDF's Part 1 verification step and for generating the signature."
>
> "Part 4's augmentation is completely different. **It happens in memory only, applied every time a training batch is built**. We only use 2 ops here — `RandomHorizontalFlip(p=0.5)` and `RandomRotation(15°)`. Not all 6 — just the two most effective ones. If you add Skew, Crop, Distortion all together, learning becomes noisy and convergence slows down."
>
> "The key part is: **the same image looks different every epoch**. For example, `Apple_rust/image (5).JPG` might be horizontally flipped + rotated 8° in epoch 1, just rotated -3° in epoch 2, flipped + 14° in epoch 5. Always a different variant. So the model sees what's effectively 25 different images over 25 epochs, while only one original is on disk. Infinite-data effect."
>
> "And **no augmentation on val**. Just resize and normalize, for honest measurement."

### 7-3. Model architecture — ScratchCNN

Open `src/leaffliction/models/scratch_cnn.py`:

🎤

> "The model has two parts. `self.features` is the feature extractor — 4 conv blocks stacked. `self.head` is the classifier — GAP + Dropout + Linear."
>
> "One conv block is 7 layers: `Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2)`."

**Conv2d explanation**:

> "`Conv2d` uses small learnable 3×3 filters to scan the image and produce new channels. Channels are a bit abstract — think of one channel as a grayscale map showing the same image from one perspective. Input is RGB so 3 channels. After the first conv block it becomes 32 channels, then 64, 128, 256. Each channel shows where a different pattern is in the image, and what those patterns are is learned automatically during training."

**BatchNorm / ReLU / MaxPool**:

> "BatchNorm normalizes a layer's output to mean 0 and variance 1. It stabilizes training, and even if input brightness or contrast varies, the model stays consistent."
>
> "ReLU is the function `max(0, x)`. Negative values become 0, positives stay. It introduces non-linearity, which is essential. Without ReLU, no matter how many layers you stack, you end up with a single linear model — a straight line."
>
> "MaxPool keeps the maximum value in each 2×2 region. Spatial resolution halves: 256 → 128 → 64 → 32 → 16. We discard fine detail and focus on the bigger picture."
>
> "After 4 blocks, input `(3, 256, 256)` becomes `(256, 16, 16)`. Channels grow as spatial resolution shrinks — the classic funnel structure of CNNs."

**Head**:

> "The head has 4 layers. `AdaptiveAvgPool2d(1)` is GAP — Global Average Pooling. It takes the `(256, 16, 16)` from the last conv and averages spatially to `(256,)`, a 256-dimensional vector. It compresses to 'how strong is each of the 256 patterns across the whole image.' This solves the old problem of huge FC layers and gives location invariance for free."
>
> "Dropout(0.4) randomly turns off 40% of neurons during training. This prevents specific neurons from memorizing the answer and helps with overfitting. It's automatically off during evaluation."
>
> "Finally `Linear(256, 8)` produces 8 class logits. Apply softmax to get probabilities. Total parameters around 1.18 million."

### 7-4. Training loop — Forward, Backward, Backprop

Open `src/leaffliction/trainer.py`:

🎤

> "The core of the training loop is 5 lines of PyTorch."

```python
logits = model(x)              # ① forward
loss = criterion(logits, y)    # ② loss
optimizer.zero_grad()          # ③ clear old gradients
loss.backward()                # ④ backward (backpropagation)
optimizer.step()               # ⑤ update weights
```

> "Going line by line — **forward is making predictions with the current weights**. The input image passes through the 4 conv blocks and the head, producing logits for 8 classes."
>
> "**Loss** scores how wrong the prediction is. We use `CrossEntropyLoss` — if the correct class probability is near 1, loss is near 0; if it's far, loss is large."
>
> "**Backward, this is backpropagation, also called 역전파 in Korean**. Intuitively — the model has millions of weights, and we need to know how much each weight contributed to the loss to figure out how to adjust it. That's `∂loss/∂weight`, the derivative."
>
> "But a neural network is a composite function: `loss = L(head(conv4(conv3(conv2(conv1(x))))))`. You can't directly differentiate the deepest conv1's weight with respect to loss. You have to apply **the chain rule**, going from loss back to input layer by layer. That's why it's called 'back'propagation. Forward goes input → output, backward goes loss → input, multiplying derivatives along the way."
>
> "PyTorch's **autograd** does this automatically. During forward it remembers the computation graph, then `loss.backward()` automatically applies the chain rule and fills in the `.grad` attribute of every weight. Back in the 80s people had to write these derivative formulas by hand — autograd is one of the key reasons deep learning took off."
>
> "Finally **optimizer.step()** moves each weight slightly in the opposite direction of its gradient. Something like `weight = weight - learning_rate × gradient`. This is called gradient descent."
>
> "We run these 5 lines once per batch. Batch is 32 images, train has 5,783, so about 181 steps per epoch. 25 epochs is about 4,500 steps total. Weight gets nudged toward the correct answer 4,500 times during training."

### 7-5. Loss / Optimizer / Scheduler / Early stop — and why epochs=25

🎤

> "Loss is the `CrossEntropyLoss` I mentioned."
>
> "Optimizer is `Adam(lr=1e-3, weight_decay=1e-4)`. Adam adapts the step size per weight — combines momentum and RMSprop, converges faster than plain SGD. `weight_decay` is L2 regularization that prevents weights from getting too large, helps with overfitting."
>
> "Scheduler is `ReduceLROnPlateau(factor=0.5, patience=2)`. If val_accuracy stalls for 2 epochs, the learning rate halves. Useful for fine-tuning in late training."
>
> "Early stopping uses `patience=5`. If val_acc doesn't improve by 0.1% for 5 epochs, training stops automatically. It's a safety net to stop right before overfitting kicks in."

If they ask about epochs=25:

> "`--epochs 25` is an upper bound, not a target. In practice we rarely actually run 25 epochs — early stopping usually kicks in around epoch 17-22."
>
> "The reason is, this is a clear dataset and learning converges quickly. Usually around epoch 14-17 we hit best val_acc of 99.8%, and after that no more 0.1%+ improvement happens. With patience=5, we stop 5 epochs after the best — so termination around 19-22 is natural."
>
> "25 itself — too low (like 10) risks underfitting, too high (like 100) wastes time if early stopping doesn't trigger. ML rule of thumb is that clear classification problems converge in about 20 epochs, so 25 has a bit of margin."

> "One side note — because `WeightedRandomSampler(replacement=True)` is used, not every image is guaranteed to appear in one epoch. For Apple_healthy specifically, about 45% of images might not show up in a single epoch. But cumulatively over 25 epochs, every image is almost certainly seen. And each appearance has a different random transform, so effectively each image is exposed 14-80 times. Learning load is more than enough."

### 7-6. Result visualization

Open `artifacts/learning_curves.png`:

🎤

> "These are the learning curves. train_loss and val_loss decrease together and converge. You'll notice some epochs have train_loss higher than val_loss — that might look weird but it's actually normal. We apply augmentation only to train batches, so train is a harder problem than val. **This is the opposite signal of overfitting** — it's evidence that augmentation is working."

Open `artifacts/confusion_matrix.png`:

> "Confusion matrix. The diagonal is almost entirely filled, off-diagonals are 1-2 images. 4 out of 8 classes are 100%, the rest are 99% range."

Open `artifacts/classification_report.txt` and `artifacts/metadata.json`:

> "scikit-learn's classification report shows precision, recall, f1 per class. `metadata.json` has best_epoch, val_accuracy, class layout. That file goes into `trained_models.zip` so `predict.py` can map class labels."

### 7-7. Data leakage avoidance — the most common follow-up

🎤

> "With 99.8% accuracy you might suspect overfitting or leakage. The key is **the order of augmentation and split**."
>
> "The wrong order is augment-then-split. If you split the `augmented_directory/` directly, variants of the same original — `image (1).JPG` and `image (1)_Flip_0.JPG` — get scattered between train and val. The model sees the answer it memorized from train almost-identically in val, and you get a fake 100%. That's data leakage."
>
> "Our order is split-then-augment. **We only split the original `images/`**, and augmentation is applied to train batches only, in memory. Variants are never on disk, so there's no path for them to leak into val. Zero risk of leakage."
>
> "Actually, when I tried v1 trained on `augmented_directory`, it gave 100% — suspicious. I switched v2 to original + online augmentation, and it dropped naturally to 99.79%. That's the honest result."

Transfer comparison:

> "And as a comparison, we also have EfficientNet-B0 transfer learning. EfficientNet-B0 is pretrained on ImageNet's 1 million images. The early conv layers learn to detect edges, curves, textures — that knowledge transfers to leaf images. We just replace the last 1000-class classifier with an 8-class one and fine-tune in two stages. Epochs 1-5 freeze the backbone and train only the classifier; epoch 6 onward we unfreeze everything with 1/10 the learning rate for fine-tuning. The result is 99.86%, nearly identical to ScratchCNN. Both models reaching similar accuracy is another sign that the dataset is clear."

### 📂 Related code (entire section)
- `src/leaffliction/models/scratch_cnn.py` — ScratchCNN: 4 conv blocks + GAP head.
- `src/leaffliction/models/transfer.py` — EfficientNet-B0 wrapper with `freeze()` / `unfreeze()`.
- `src/leaffliction/trainer.py` — `train()` function with the PyTorch 5-line loop, optimizer, scheduler, early stop, two-stage fine-tune branch.
- `src/leaffliction/dataset.py::LeafDataset` — PyTorch Dataset. Transform applies fresh every `__getitem__` → online augmentation.
- `train.py` — typer CLI. Stratified split, WeightedRandomSampler setup, model dispatch, result visualization, zip + signature.

---

## 8. Part 4 (3/4) — Unit_test1 (Apple)

### 🎯 What the evaluator checks
10 images in `test_images/Unit_test1/`, one point each for correct prediction. They may rename files to prevent cheating.

### 💻 Commands
```bash
# whole folder at once — multi-mode with auto self-check
./predict.py /tmp/test_images/Unit_test1/

# save PNGs too
./predict.py /tmp/test_images/Unit_test1/ --save /tmp/out_unit1/

# single image — matches the PDF example
./predict.py /tmp/test_images/Unit_test1/Apple_healthy1.JPG
```

### 🎤 What to say

> "Unit_test1 is 10 images from the 4 Apple classes. If I pass a folder to `predict.py`, it auto-collects all `*.JPG` inside and predicts them all at once. If I give it a single image, it shows the matplotlib figure like the PDF example. If it's a folder, it outputs a clean console table."

Output example:

```
Predicting 10 images with model=scratch...
  OK   Apple_healthy      (99.8%)  ← Apple_healthy1.JPG
  OK   Apple_Black_rot    (99.1%)  ← Apple_BlackRot2.JPG
  ...
Self-check: 10/10 = 100.00%
```

> "If the filename starts with a class name — like the PDF example `Apple_healthy1.JPG` — auto self-check kicks in. If the evaluator renames files, self-check just skips and shows blank in the marker column, but prediction still works. **The filename is not used as prediction input** — it's just auxiliary info for self-check. The model reads class labels from `metadata.json` inside the zip and predicts using only the image pixels."

If they ask about 100% confidence:

> "About the 100.0% confidence values — that's the softmax output, not literal 100% probability. It means the model is very confident. Neural networks tend to be overconfident once they're well-trained, and softmax's exponential pushes the largest logit very close to 1.0 even when other logit differences are moderate. With wild outdoor photos you'd probably see 70-95% instead."

### 📂 Related code
- `predict.py` — typer CLI. Auto-dispatches: single file → figure mode, multi / directory → console table. `_guess_class_from_name` uses filename prefix match for self-check.
- `src/leaffliction/predictor.py`
  - `load_artifact(zip, prefer)` — unzip + load model + classes once. Used once across N images in multi mode.
  - `predict_one(artifact, image)` — PIL load → resize → normalize → forward → softmax argmax.
  - `render(result, save)` — 2-panel figure (original + mask transform).

---

## 9. Part 4 (4/4) — Unit_test2 (Grape)

### 🎯 What the evaluator checks
10 Grape images. **If all 10 are wrong, the rubric explicitly tells the evaluator to suspect data leakage.**

### 💻 Commands
```bash
./predict.py /tmp/test_images/Unit_test2/
./predict.py /tmp/test_images/Unit_test2/ --save /tmp/out_unit2/
```

### 🎤 What to say

> "Unit_test2 is 10 Grape images. Same folder multi-mode."
>
> "The rubric explicitly says 'if all 10 are wrong, suspect how the student got high validation accuracy.' That's pointing at leakage suspicion. As I explained in 7-7, we kept split-then-augment order to block leakage, and Unit_test2 passing is additional evidence that 99.79% on val is a legitimate result."

### ⚠️ Pitfalls
All 10 wrong → leakage suspicion. Restate the 7-7 leakage avoidance argument.

---

## 10. Common Questions (Q&A)

### Q1. Why did you write CNN from scratch? Isn't transfer learning easier?
> "I have both. Default is ScratchCNN. Two reasons. First, I designed every layer myself so I can explain them one by one — that's safer for defense. Second, the rubric gives 5 points for model explanation, so a hand-designed model is more defensible than a black-box pretrained one. Transfer is included as comparison and to show how it's used in production."

### Q2. Why EfficientNet-B0 specifically?
> "It's the modern standard baseline — best parameter/compute efficiency at a given accuracy. ResNet-50 is 25M params, B0 is 5M — 1/5. Fast inference even on CPU."

### Q3. What exactly is data leakage?
> "It's when the model has seen information at training time that it shouldn't see at evaluation. The most common case is augment-then-split. Variants of the same original end up in both train and val, so the model meets nearly-identical images in val that it memorized in train — gives a fake 100%. We avoid this by splitting originals only and applying augmentation only in memory at batch time."

### Q4. Why not use augmented_directory for training and skip WeightedRandomSampler?
> "`augmented_directory` exists for PDF requirements but training on it would cause leakage. So we needed another way to balance classes, and that's `WeightedRandomSampler`. It achieves the same effect at batch level without disk duplication — no leakage."

### Q5. Why seed=42?
> "The number itself is meaningless — I just used it because it's a common ML convention. What matters is that it's fixed. 0, 12345, anything works. The point is that train.py and eval_val.py both use the same seed so the val split is reproducible."

### Q6. 99.8% accuracy seems too good to be real.
> "Three pieces of evidence. First, this dataset has controlled gray backgrounds and only one leaf per image, so visual differences between classes are very clear — it's an easy problem for a model. Second, ScratchCNN and EfficientNet — two completely different architectures — get 99.79% and 99.86%, within 0.07pp. So it's not a trick of one specific model, it's the dataset being clear. Third, the confusion matrix is natural, with 1-2 misclassified per class scattered around. And `eval_val.py` can re-verify immediately right here."

### Q7. How do you prevent overfitting?
> "Five things at once. Dropout 0.4, weight_decay 1e-4, online augmentation, early stopping with patience=5, ReduceLROnPlateau. The fact that train_loss and val_loss converge together in `learning_curves.png` is evidence — and sometimes train_loss is even higher than val_loss because we only augment train batches."

### Q8. Why epochs=25?
> "It's an upper bound. For clear classification problems, ML rule of thumb is convergence around 20 epochs, plus patience=5 margin gave me 25. In practice early stopping kicks in around epoch 17-22, so we usually don't run all 25."

### Q9. What's forward and backward?
> "Forward is making a prediction with the current weights. Backward is backpropagation — using the chain rule from the output back to the input, layer by layer, to compute how each weight contributed to the loss. Layers are composite functions, so we need to apply derivatives backward through the chain. PyTorch's autograd handles this automatically — we only define forward, and `loss.backward()` computes every weight's gradient in one line."

### Q10. Why uv?
> "It's a modern Python tool that combines pip, virtualenv, pyenv, and pip-tools. Written in Rust, 10-100× faster than pip, and `uv.lock` guarantees dependency reproducibility."

### Q11. What if you don't know the answer?
> "I'd say 'honestly, not sure — let me check the code with you,' and open the file. Honesty is safer for scoring. The rubric's zero-point condition is 'can't explain at all,' not 'doesn't know one or two specifics.'"

---

## 11. Risk Scenarios + Responses

### `make verify` fails (signature mismatch)
```bash
ls -la trained_models.zip augmented_directory.zip
shasum trained_models.zip augmented_directory.zip
# → try backup USB. Last resort: ./train.py images/ --epochs 25 (~50 min)
```

### `uv sync` fails (plantcv build error)
```bash
uv pip install plantcv --no-build-isolation
# or
python -m pip install -e .
```

### matplotlib window doesn't appear (SSH/Docker)
```bash
MPLBACKEND=Agg ./Distribution.py images/ --save /tmp/dist.png
open /tmp/dist.png
```

### All Unit_test predictions are wrong
Restate the 7-7 leakage avoidance argument + show `artifacts/confusion_matrix.png` as evidence that the model works on the internal distribution. Acknowledge that out-of-distribution data (like outdoor phone photos) may naturally lower accuracy.

### `./Distribution.py: command not found`
```bash
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py
# or check venv activation
source .venv/bin/activate
# or use uv directly
uv run python Distribution.py ./images
```

### Evaluator gives a different dataset path
All entrypoints accept a path argument — not tied to our `images/`.
```bash
./Distribution.py /evaluator/path
./predict.py "/evaluator/Unit_test1/"
```

---

## 12. Closing One-Liner

> "We passed all 5 PDF entrypoints and the rubric's zero-point traps — augmented_directory balance, signature.txt match, 100+ images with ≥ 90% accuracy — and the result is honest because we blocked data leakage. ScratchCNN is hand-designed so we can explain every layer, and the EfficientNet-B0 transfer model is also there as a comparison to show how CNN is deployed in production."

---

## 📚 Appendix: Detailed Code Walkthrough

This appendix isn't for reading aloud during the defense — it's for **study before the defense**, to understand how the code actually works. Each Part is walked through in execution order, so that when you come back to the code later you can say "ah, this is what this is."

### Overall file map (one more time before the appendix)

```
User runs a command (e.g., ./Distribution.py ./images)
         ↓
Root entrypoint (.py file) — typer parses CLI args
         ↓
src/leaffliction/<module>.py — actual logic
         ↓
matplotlib display, or file written to disk
```

The 5 entrypoints at the root are **thin wrappers**. The real work is in the modules under `src/leaffliction/`.

---

### A. Distribution.py flow

#### A.1 File map

```
Distribution.py
    │  ① typer parses CLI args (directory, --save)
    │  ② die() — graceful exit on error
    │
    ├──→ dataset.py::discover_classes(directory)
    │       └─ inspects folder structure, returns {class_name: [image paths]}
    │
    └──→ viz.py::pie_and_bar(counts, title, save)
            └─ matplotlib draws two side-by-side panels
```

#### A.2 From user command to chart appearing

```
User: ./Distribution.py ./images
   ↓
[1] typer converts "./images" string into a Path object
   ↓
[2] main() is called with:
       directory = Path("./images")
       save = None
   ↓
[3] discover_classes(directory) is called
       ↓
       Does ./images directly contain images? → No (only subfolders)
       ↓
       Iterate subfolders: Apple_Black_rot, Apple_healthy, Apple_rust, ...
       ↓
       Gather *.jpg files in each subfolder (sorted)
       ↓
       Return: {
         "Apple_Black_rot": [Path("images/Apple_Black_rot/image (1).JPG"), ...],
         "Apple_healthy":   [Path("images/Apple_healthy/image (1).JPG"), ...],
         ... (8 classes)
       }
   ↓
[4] main computes counts:
       counts = {name: len(paths) for name, paths in classes.items()}
              = {"Apple_Black_rot": 621, "Apple_healthy": 1640, ...}
   ↓
[5] pie_and_bar(counts, title="images", save=None) is called
       ↓
       fig, (ax_pie, ax_bar) = plt.subplots(1, 2)
       ↓
       ax_pie.pie(...)
       ax_bar.bar(...)
       ↓
       Number labels above each bar via ax_bar.text(...)
       ↓
       plt.show() ← matplotlib window opens
```

#### A.3 Why `discover_classes` is clever

```python
def discover_classes(root):
    direct_images = _images_in(root)
    if direct_images:
        return {root.name: direct_images}      # Layout 1
    
    classes = {}
    for child in sorted(...):                  # Layout 2 (our standard)
        child_images = _images_in(child)
        if child_images:
            classes[child.name] = child_images
            continue
        for grand in sorted(...):              # Layout 3 (nested)
            ...
```

It handles 3 folder layouts:

```
Layout 1 (single folder — like passing Apple_healthy/ directly):
  Apple_healthy/
    ├── image (1).JPG
    └── ...
  → {"Apple_healthy": [Path, Path, ...]}

Layout 2 (our standard — like images/):
  images/
    ├── Apple_healthy/
    └── ... (8 classes)
  → {"Apple_healthy": [...], "Apple_Black_rot": [...], ...}

Layout 3 (extra group wrapper — like Apple/ containing healthy/, scab/...):
  Apple/
    ├── apple_healthy/
    └── apple_scab/
  → {"apple_healthy": [...], "apple_scab": [...]}
```

This way, whatever folder structure the evaluator provides, the code still works.

**Why `sorted()` in two places**: same folder structure → same class indexing every time. So train.py's class indices (0=Apple_Black_rot, 1=Apple_healthy, ...) match eval_val.py's. **The starting point of reproducibility.**

---

### B. Augmentation.py flow

#### B.1 File map

```
Augmentation.py
    │  ① target.is_file() ─→ single mode
    │  ② target.is_dir()  ─→ batch mode
    │
    ├─→ [single] augment.py::apply_op + save_with_suffix
    │       └─ apply all 6 ops, save siblings
    │       and show via viz.py::grid
    │
    └─→ [batch] augment.py::balance_directory
            └─ augment each class up to target_count
            └─ zip_directory creates the .zip
```

#### B.2 Single mode (for the PDF demo)

```
User: ./Augmentation.py "images/Apple_healthy/image (1).JPG"
   ↓
[1] target = Path("images/Apple_healthy/image (1).JPG")
       target.is_file() → True → single mode
   ↓
[2] rgb = load_image(target)
       └─ PIL.Image.open(target).convert("RGB") → numpy (H, W, 3)
   ↓
[3] outputs = [("Original", rgb)]    # for grid display
   ↓
[4] Iterate 6 ops:
   for name in AUGMENTATION_OPS:                # ["Flip", "Rotate", "Skew", ...]
       aug = apply_op(name, rgb)                # transformed numpy
       save_with_suffix(target, aug, name)      # writes image (1)_Flip.JPG etc
       outputs.append((name, aug))              # also for grid
   ↓
[5] grid(outputs)
       └─ matplotlib subplots 1×7 (original + 6 variants)
       └─ ax.imshow(img) + set_title(label)
```

`AUGMENTATION_OPS` is a dict so iteration order is deterministic (Python 3.7+'s insertion order): Flip → Rotate → Skew → Shear → Crop → Distortion.

#### B.3 Batch mode (for Part 1 verify)

```
User: ./Augmentation.py images/
   ↓
[1] target.is_dir() → True → batch mode
   ↓
[2] balance_directory(target, output="augmented_directory", ...)
       ↓
       [a] discover_classes(src) collects image paths per class
       [b] target = max(class lengths) = 1640 (Apple_healthy)
       ↓
       [c] For each class:
           for cls, paths in classes.items():
               # copy originals
               for src in paths[:target]:
                   shutil.copy2(src, dst/<cls>/<filename>)
               produced = number copied
               
               # augment until target reached
               while produced < 1640:
                   base = rng.choice(paths)             # random pick within class
                   img = load_image(base)
                   op_name, aug = apply_random_op(img)  # random 1 of 6 ops
                   save: <stem>_<op>_<counter>.JPG
                   produced += 1
       ↓
       [d] When done, zip_directory(dst) auto-fires
           └─ creates augmented_directory.zip with ZIP_DEFLATED
```

Visualized class-balance result:

```
Original (imbalanced)        After (balanced)
─────────────                ─────────────
Apple_healthy   1640         Apple_healthy   1640  (copied as-is)
Apple_Black_rot  621         Apple_Black_rot 1640  (621 copied + 1019 augmented)
Apple_rust       275         Apple_rust      1640  (275 copied + 1365 augmented)
Apple_scab       630         Apple_scab      1640  (630 copied + 1010 augmented)
... (8 classes)              ... (8 × 1640 = 13,120 images)
```

#### B.4 The one-line core of `apply_op`

```python
def apply_op(name, image):
    transform = AUGMENTATION_OPS[name]
    return transform(image=image)["image"]
```

Albumentations API: calling `transform(image=array)` returns `{"image": transformed_array, ...}` dict. Just extract the `"image"` key. The function body is one line. Thanks to dict structure, the 6 ops can be expressed as a dict + one-line call.

---

### C. Transformation.py flow

#### C.1 File map

```
Transformation.py
    │  ① image arg → single mode (one figure with 6 transforms + hist)
    │  ② -src/-dst args → batch mode (whole directory)
    │
    └─→ transform.py::
            load_rgb         (PIL → numpy)
            _binary_mask     (5-stage mask pipeline — core)
            all_transforms   (returns dict of 6 transforms)
            color_histogram  (9-channel hist)
```

#### C.2 Single mode flow

```
User: ./Transformation.py "images/Apple_healthy/image (1).JPG"
   ↓
[1] rgb = load_rgb(image)             # (H, W, 3) numpy uint8
   ↓
[2] outs = all_transforms(rgb)        # 6 transforms in one call
       returns: {"Original": ..., "GaussianBlur": ..., "Mask": ...,
                 "RoiObjects": ..., "AnalyzeObject": ..., "Pseudolandmarks": ...}
   ↓
[3] hist = color_histogram(rgb)       # 9-channel histogram
       returns: {"red": array(256,), "green": ..., ..., "blue-yellow": ...}
   ↓
[4] matplotlib figure layout:
       
       fig = plt.figure(figsize=(16, 9))
       
       ┌─────┬─────┬─────┬─────┬─────┬─────┐
       │ Org │ Blur│ Mask│ ROI │Analy│ Land│   ← subplot2grid((2,6), (0, i))
       ├─────┴─────┴─────┴─────┴─────┴─────┤
       │       9-channel color histogram     │   ← subplot2grid((2,6), (1,0), colspan=6)
       └───────────────────────────────────┘
   ↓
[5] Each top cell: ax.imshow(img) (2D arrays get cmap="gray", vmin=0, vmax=255)
[6] Bottom cell: plot all 9 channels in one axes + legend
[7] plt.show()
```

Why `vmin=0, vmax=255` matters: matplotlib auto-normalizes images with mostly bright pixels (like masks), making them look dark gray. Explicit 0-255 range prevents this.

#### C.3 `_binary_mask` 5-stage pipeline visualized

```
Input: rgb (H, W, 3)  ─ healthy apple leaf + gray background
   ↓
[1] cv2.cvtColor(rgb, RGB2LAB) → lab (H, W, 3)
       a = lab[..., 1] - 128       ← green↔red axis, gray near 0
       b = lab[..., 2] - 128       ← blue↔yellow axis, gray near 0
       chroma = sqrt(a² + b²)       ← 0~255 (gray=0, green/brown=large)
   ↓
   chroma visualization (gray=dark, colored=bright):
   ┌─────────────┐
   │ ░░░░░░░░░░░ │   ← gray background (chroma ≈ 0)
   │ ░░██████░░░ │   ← leaf (chroma ≈ 100)
   │ ░░██▓▓██░░░ │   ← brown lesion (chroma ≈ 80)
   │ ░░░░░░░░░░░ │
   └─────────────┘
   ↓
[2] cv2.threshold(chroma, 0, 255, BINARY+OTSU)
       Otsu picks a cutoff automatically (e.g. 30) and binarizes
   ↓
   binary:
   ┌─────────────┐
   │ ░░░░░░░░░░░ │   ← 0 (background)
   │ ░░████████░ │   ← 255 (leaf)
   │ ░░████████░ │
   │ ░░░░░░░░░░░ │
   └─────────────┘
   ↓
[3] cv2.morphologyEx(binary, MORPH_OPEN, kernel=3×3)
       (erosion + dilation) — removes small noise specks
   ↓
[4] cv2.connectedComponentsWithStats
       Finds all "white blobs" in binary, keeps only the largest
       → removes any remaining small scattered blobs
   ↓
[5] scipy.binary_fill_holes
       The leaf outline is captured but small black holes remain inside
       → fills the interior with white
   ↓
[6] pcv.fill(bin_img=filled, size=200)
       Removes any speck under size 200 (extra safety margin)
   ↓
Output: binary mask (H, W) ─ 255 for leaf, 0 for background
```

This mask is the **shared preprocessing for `gaussian_blur`, `mask`, `roi_objects`, `analyze_object`, `pseudolandmarks` — 5 of the 6 functions**. Get the mask right once, everything else looks clean.

#### C.4 How the 9-channel histogram is built

```python
def color_histogram(rgb):
    hsv = cv2.cvtColor(rgb, RGB2HSV)       # color space conversion 1
    lab = cv2.cvtColor(rgb, RGB2LAB)       # color space conversion 2
    
    channels = {                            # 9 channels bundled as dict
        "blue":  rgb[..., 2], "green":      rgb[..., 1], "red":            rgb[..., 0],
        "hue":   hsv[..., 0], "saturation": hsv[..., 1], "value":          hsv[..., 2],
        "lightness": lab[..., 0], "green-magenta": lab[..., 1], "blue-yellow": lab[..., 2],
    }
    
    out = {}
    total = H * W                           # total pixel count
    for name, ch in channels.items():
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        out[name] = 100.0 * hist / total    # normalize to % ← makes images of different sizes comparable
    return out
```

Data flow:

```
rgb (H, W, 3)                   ── one image
     ↓
[color space conversion]
RGB (used directly) ─── R, G, B each (H, W) — 3 channels
HSV (converted)     ─── H, S, V each (H, W) — 3 channels
LAB (converted)     ─── L, a*, b* each (H, W) — 3 channels
     ↓ (9 (H,W) arrays)
[np.histogram(bins=256) for each channel]
     ↓
9 arrays of length 256 (pixel value frequencies)
     ↓
[normalize to %]
     ↓
{"red": array(256,), "green": ..., ..., "blue-yellow": ...}
```

---

### D. train.py flow (most complex, most important)

#### D.1 Big picture

```
User: ./train.py images/ --epochs 25
   ↓
[Phase 1: prep]
   set_seed(42) ──── all RNGs fixed (Python/NumPy/PyTorch/cuDNN)
   ↓
   _build_loaders() ── builds train/val DataLoaders
       │
       ├─ LeafDataset(images, train_tf) ─── 5,783 samples
       ├─ LeafDataset(images, val_tf)   ─── 1,445 samples (same folder, different transform)
       ├─ train_test_split(stratify, seed=42) ─── 80/20 index split
       ├─ WeightedRandomSampler ─── weight = 1/class_size
       └─ DataLoader (num_workers=8, persistent_workers=True)
   ↓
[Phase 2: training — 25 epoch upper bound, early stop terminates]
   _train_one("scratch", ScratchCNN(), train_loader, val_loader, TrainConfig)
       │
       └─ trainer.py::train()
              │
              for epoch in 1..25:
                  ├─ _epoch(model, train_loader, criterion, optimizer)  ← training
                  │      └─ forward/loss/backward/step per batch
                  ├─ _epoch(model, val_loader, criterion, None)          ← eval (no gradient)
                  ├─ scheduler.step(val_acc)
                  ├─ best updated? → save best_state
                  └─ 5 epoch plateau? → break (early stop)
   ↓
[Phase 3: finalize]
   build confusion matrix using best model
   save metadata.json + learning_curves.png + confusion_matrix.png
   ↓
   zip artifact:
       trained_models.zip ← compress entire out/ folder
   ↓
   write signature.txt (SHA1 of zips)
```

#### D.2 `_build_loaders` step by step

The function with the most crucial tricks. Worth reading carefully.

```python
def _build_loaders(directory, split, batch, seed):
    
    # ① Two transforms — train only gets augmentation
    train_tf = Compose([
        Resize((256, 256), antialias=True),
        RandomHorizontalFlip(p=0.5),        # ← 50% flip each call
        RandomRotation(degrees=15),          # ← random -15..+15° each call
        Normalize(mean=ImageNetMean, std=ImageNetStd),
    ])
    val_tf = Compose([
        Resize((256, 256), antialias=True),
        Normalize(mean=ImageNetMean, std=ImageNetStd),
        # ← no augmentation!
    ])
```

**Trick 1**: train and val need different transforms, so we build LeafDataset twice.

```python
    # ② LeafDataset twice — same folder, different transforms
    train_full = LeafDataset(directory, transform=train_tf)
    val_full = LeafDataset(directory, transform=val_tf)
    labels = [lab for _, lab in train_full.samples]
```

Data structure visualization:

```
train_full.samples = [
    (Path("images/Apple_Black_rot/image (1).JPG"), 0),  # idx 0
    (Path("images/Apple_Black_rot/image (2).JPG"), 0),  # idx 1
    ...
    (Path("images/Apple_healthy/image (1).JPG"),  1),
    ...
    (Path("images/Grape_spot/image (1076).JPG"),  7),   # idx 7227 (last)
]
val_full.samples = [exactly the same 7,228 (path, label)]  ← discover_classes sorts so order matches
```

```python
    # ③ Stratified split — same indices apply to both datasets
    train_idx, val_idx = train_test_split(
        list(range(len(labels))),            # [0, 1, 2, ..., 7227]
        test_size=1-split,                    # 0.2
        stratify=labels,                      # preserve class ratios
        random_state=seed,                    # 42
    )
    # train_idx: [3, 12, 47, ...] (5,783)
    # val_idx:   [0, 5, 23, ...]  (1,445)
    
    train_ds = Subset(train_full, train_idx)  # gets train_tf
    val_ds = Subset(val_full, val_idx)        # gets val_tf
```

`Subset` is a lightweight view over the original Dataset with an index list — no data copy.

**Trick 2**: because train_full and val_full have the same sample order, the indices from train_test_split point to the same images in both. Only the applied transform differs.

```python
    # ④ WeightedRandomSampler — class balance
    train_labels = np.array([labels[i] for i in train_idx])
    class_count = np.bincount(train_labels)
    # array([ 497, 1312,  220,  504,  944, 1106,  339,  861])
    
    sample_weights = 1.0 / class_count[train_labels]
    # length 5,783, each weight = 1/(its class size)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),   # 5,783 — picks per epoch
        replacement=True,                   # duplicates OK
    )
```

**Calculation walkthrough**:

```
If sample at train_idx[0] is Apple_rust (class_count[2]=220):
    sample_weights[0] = 1/220 = 0.00455

If sample at train_idx[1] is Apple_healthy (class_count[1]=1312):
    sample_weights[1] = 1/1312 = 0.000762

→ Apple_rust gets picked 6× more often than Apple_healthy
→ Total weight per class = 1 (8 classes × 1 = 8.0 total)
→ Each batch has nearly equal class representation
```

```python
    # ⑤ DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch,        # 32
        sampler=sampler,                    # the WeightedRandomSampler from above
        num_workers=8,                      # 8 background workers prefetch batches
        persistent_workers=True,            # don't recreate workers every epoch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch,
        shuffle=False,                      # val doesn't need shuffling
        num_workers=8, persistent_workers=True,
    )
```

`num_workers=8` matters. macOS has expensive fork, so recreating workers per epoch costs a lot. `persistent_workers=True` keeps them alive across epochs.

#### D.3 What happens when `LeafDataset.__getitem__` is called (the online augmentation secret)

DataLoader builds a batch internally like this:

```
DataLoader(batch_size=32)
   ↓
sampler picks 32 indices weighted: [47, 1893, 47, 502, 1, 47, ...]
   ↓                                              ↑↑↑↑ duplicates OK!
8 workers in parallel call dataset[<idx>]:
   dataset[47]
       ↓
       train_ds.__getitem__(47)
           ↓
           Subset translates to train_full.__getitem__(train_idx[47])
               ↓
               # LeafDataset.__getitem__:
               path, label = self.samples[idx]
               img = Image.open(path).convert("RGB")          # load from disk
               tensor = F.to_image(img)                        # PIL → tensor (3, H, W) uint8
               tensor = F.to_dtype(tensor, float32, scale=True) # uint8 → float32 [0,1]
               
               if self.transform is not None:
                   tensor = self.transform(tensor)             # ← train_tf applied
                                                                 # fresh random each time!
               return tensor, label
   ↓
8 workers bring back 32 (tensor, label) tuples
   ↓
collate: stack 32 tensors on dim 0 → (32, 3, 256, 256)
         stack 32 labels → (32,)
   ↓
DataLoader yields (x, y) — training loop receives it
```

**Why the same image looks different every epoch**:

```
Say sampler picks index 47 in both epoch 1 and epoch 2.

Epoch 1 call:
    __getitem__(47) → load img → train_tf(img)
                                       ↓
                                  RandomHorizontalFlip.forward() called
                                       ↓
                                  random.random() < 0.5? → True (flip this time)
                                       ↓
                                  RandomRotation.forward() called
                                       ↓
                                  random.uniform(-15, 15) → 8.3° (rotation this time)
                                       ↓
                                  return transformed tensor

Epoch 2 call:
    __getitem__(47) → same img loaded → train_tf(img)
                                       ↓
                                  random.random() < 0.5? → False (no flip this time)
                                       ↓
                                  random.uniform(-15, 15) → -3.1° (different angle)
                                       ↓
                                  return completely different transformed tensor
```

Key point: `RandomHorizontalFlip` and `RandomRotation` are **stateless** — they roll a fresh random value every call. So calling N times on the same sample gives N different variants. Disk holds only one original.

#### D.4 What happens inside one epoch

```
def _epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None         # train epoch?
    model.train(is_train)                     # switch BN/Dropout mode
    loss_sum, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(is_train):    # gradient on/off
        for x, y in loader:                   # iterate DataLoader
            x = x.to(device)                  # move to GPU/MPS
            y = y.to(device)
            
            logits = model(x)                 # ① forward
            loss = criterion(logits, y)       # ② loss
            
            if is_train:
                optimizer.zero_grad()         # ③ clear gradients
                loss.backward()               # ④ backward (backprop)
                optimizer.step()              # ⑤ update weights
            
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return loss_sum / total, correct / total
```

**Time-ordered view of one epoch (181 steps)**:

```
Step 1:
  batch_1 = next(train_loader)  ← (32, 3, 256, 256) + 32 labels
  x.to(device), y.to(device)    ← memory move (CPU → MPS/CUDA)
  logits = model(x)             ← (32, 8) 8-class logits
  loss = CrossEntropyLoss(logits, y)  ← scalar
  optimizer.zero_grad()
  loss.backward()               ← .grad filled for all model.parameters()
  optimizer.step()              ← weight = weight - lr * .grad
  loss_sum += loss.item() * 32
  correct += correct count
  
Step 2:
  ... same
  
...

Step 181:
  ... last batch (5,783 mod 32 = 23 leftover, sampler fills)
  
return: train_loss, train_acc
```

When the train epoch ends, the same function runs val epoch with `optimizer=None`:

```
Val epoch (46 steps):
  with torch.set_grad_enabled(False):  ← no gradient calc
      for x, y in val_loader:
          logits = model(x)
          loss = criterion(logits, y)
          # done — no backward or optimizer.step
          loss_sum += ...
          correct += ...

return: val_loss, val_acc
```

#### D.5 Forward / backward visualized

Inside the ScratchCNN, how one image's shape changes:

```
Input batch x: (32, 3, 256, 256)  ─ 32 RGB images
   │
   ▼
self.features (4 conv blocks)
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
   │ │   ... same pattern, channels 32→64                       │
   │ │   output: (32, 64, 64, 64)                                 │
   │ └──────────────────────────────────────────────────────────┘
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 3: _conv_block(64, 128)                            │
   │ │   output: (32, 128, 32, 32)                                │
   │ └──────────────────────────────────────────────────────────┘
   │
   │ ┌──────────────────────────────────────────────────────────┐
   │ │ Block 4: _conv_block(128, 256)                           │
   │ │   output: (32, 256, 16, 16)                                │
   │ └──────────────────────────────────────────────────────────┘
   │
   ▼
self.head
   │ AdaptiveAvgPool2d(1)  → (32, 256, 1, 1)    ─ 16×16 grid averaged to 1×1
   │ Flatten()             → (32, 256)          ─ 256-d vector per image
   │ Dropout(0.4)          → (32, 256)          ─ 40% off during training
   │ Linear(256, 8)        → (32, 8)            ─ class logits
   ▼
Output: logits (32, 8)
```

**Forward = top to bottom, compute logits**.

**Backward = bottom to top, compute gradients**:

```
loss (scalar) ← CrossEntropyLoss(logits, y)
   │
   │ compute ∂loss/∂logits → (32, 8)
   ▼
Linear(256, 8)        ← compute ∂loss/∂weight, ∂loss/∂bias
   │ chain rule → ∂loss/∂Linear_input → (32, 256)
   ▼
Dropout                ← remembers which neurons were off, gradient passes through
   ▼
Flatten                ← reshape reverse → (32, 256, 1, 1)
   ▼
AdaptiveAvgPool2d      ← derivative of average → (32, 256, 16, 16)
   ▼
[Block 4 backward]     ← derivatives of Conv, BN, ReLU each
   ▼
[Block 3 backward]
   ▼
[Block 2 backward]
   ▼
[Block 1 backward]
   ▼
gradient w.r.t. input (∂loss/∂x) — not used for training, just chain endpoint
```

PyTorch remembers the computation graph during forward, then `loss.backward()` automatically does all of this. Every `.grad` attribute of `model.parameters()` is filled in.

`optimizer.step()` then reads those `.grad`s:

```python
for p in optimizer.param_groups[0]['params']:
    p.data -= lr * p.grad   # Adam is more complex but this is the intuition
```

Each weight nudges in the opposite direction of its gradient. **Repeat this 4,500 times and weights move enough toward the correct answer for the model to learn.**

#### D.6 What happens after training — how the zip is built

```
[After training ends, train.py does:]

[1] Use best model for confusion matrix
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

[2] Save learning curves
       learning_curves(artifacts[best_name]["history"], out / "learning_curves.png")

[3] Build metadata.json
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

[4] Zip the entire artifacts/ folder
       artifacts/
       ├── model_scratch.pt           (1.18M params, ~5MB)
       ├── model_transfer.pt          (4M params, ~16MB)  [opt-in]
       ├── metadata.json
       ├── learning_curves.png
       ├── confusion_matrix.png
       └── classification_report.txt
       ↓
       trained_models.zip (≈20MB)

[5] Write signature.txt
       compute_sha1("trained_models.zip") → "101dd3f4..."
       compute_sha1("augmented_directory.zip") → "0506b961..."
       
       signature.txt contents:
       101dd3f43b16d60e1e827558fb4e10b19ae396cc  trained_models.zip
       0506b961d81e1941fc9ca972988164467a560646  augmented_directory.zip
```

On evaluation day, the USB carries trained_models.zip + augmented_directory.zip, and the hashes in `signature.txt` verify integrity.

---

### E. predict.py flow

#### E.1 File map

```
predict.py
    │  ① len(expanded) == 1 → single mode (show figure)
    │  ② len(expanded) > 1  → multi mode (console table)
    │
    └─→ predictor.py::
            load_artifact   (unzip + load model — once)
            predict_one     (inference for one image)
            render          (matplotlib figure)
```

#### E.2 Single vs multi auto-dispatch

```
User: ./predict.py "images/Apple_rust/image (1).JPG"
   ↓
[1] paths = [Path("images/Apple_rust/image (1).JPG")]
   ↓
[2] _expand(paths)
       Check each path:
         is_dir() → True: expand via rglob("*.JPG")
         is_dir() → False: append as-is
       Result: expanded = [Path("images/Apple_rust/image (1).JPG")]  ← 1 item
   ↓
[3] len(expanded) == 1 → single mode
   ↓
[4] artifact = load_artifact("trained_models.zip", prefer="scratch")
       (details in E.3)
   ↓
[5] result = predict_one(artifact, expanded[0])
   ↓
[6] render(result, save=None)
       → shows matplotlib figure
```

vs.

```
User: ./predict.py /tmp/test_images/Unit_test1/
   ↓
[1] paths = [Path("/tmp/test_images/Unit_test1/")]
   ↓
[2] _expand:
       Unit_test1/ is a directory → rglob("*.JPG") → expands 10 paths
   Result: expanded = [Apple_healthy1.JPG, Apple_BlackRot2.JPG, ...]  ← 10 items
   ↓
[3] len(expanded) > 1 → multi mode
   ↓
[4] artifact = load_artifact(...)   ← load model only once!
   ↓
[5] for path in expanded:
       result = predict_one(artifact, path)
       true_cls = _guess_class_from_name(path.name, artifact.classes)
       (prints "OK   Apple_healthy (99.8%)  ← Apple_healthy1.JPG" to console)
       if save_dir: render(..., save=save_dir / f"{path.stem}_pred.png")
   ↓
[6] Self-check: 10/10 = 100.00% shown
```

#### E.3 `load_artifact` — unzip + load model

```python
def load_artifact(zip_path, prefer="scratch"):
    # ① unzip into a hidden sibling directory
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
    
    # ② read metadata (class labels!)
    metadata = json.loads((extract_dir / "metadata.json").read_text())
    classes = metadata["classes"]  # ["Apple_Black_rot", "Apple_healthy", ...]
    
    # ③ find weight file (with fallback)
    weight_file = extract_dir / f"model_{prefer}.pt"
    if not weight_file.exists():
        for alt in ("scratch", "transfer"):
            cand = extract_dir / f"model_{alt}.pt"
            if cand.exists():
                weight_file = cand
                prefer = alt
                break
    
    # ④ build empty model and load weights
    model = _build_model(prefer, num_classes=len(classes))
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()  # ← switch BN/Dropout to eval mode (important!)
    
    return LoadedArtifact(model=model, classes=classes, image_size=256, model_used=prefer)
```

`model.eval()` is important. BN and Dropout behave differently between training and eval. Forgetting eval mode makes results non-deterministic.

#### E.4 `predict_one` — infer one image

```python
def predict_one(artifact, image_path):
    # ① load image
    rgb = np.array(Image.open(image_path).convert("RGB"))  # (H, W, 3) uint8
    
    # ② preprocess — must match train.py's val_tf exactly!
    tensor = _preprocess(rgb, size=artifact.image_size)
    # inside _preprocess:
    #   img = Image.fromarray(rgb).resize((256, 256), BILINEAR)
    #   t = F.to_image(img)                              # (3, 256, 256) uint8
    #   t = F.to_dtype(t, float32, scale=True)           # [0, 1]
    #   t = F.normalize(t, mean=ImageNetMean, std=ImageNetStd)
    #   return t.unsqueeze(0)                             # (1, 3, 256, 256) — add batch dim
    
    # ③ inference (no gradient)
    with torch.no_grad():
        logits = artifact.model(tensor)        # (1, 8)
        probs = torch.softmax(logits, dim=1)[0]  # (8,) sums to 1.0
        idx = int(probs.argmax())              # largest index
    
    return {
        "class": artifact.classes[idx],       # "Apple_rust" etc
        "confidence": float(probs[idx]),       # 0.998 etc
        "rgb": rgb,                            # original for figure
        "transformed": mask_transform(rgb),    # mask for figure
        "model_used": artifact.model_used,    # "scratch" or "transfer"
    }
```

**Why consistency with train.py matters**: if training distribution and inference distribution differ, the model receives semantically different inputs. Same resize method (BILINEAR), same normalize stats (ImageNet mean/std), or accuracy drops.

#### E.5 Tracking one image's transformation end-to-end

```
Disk: "Apple_rust1.JPG"
   ↓ Image.open + convert("RGB")
PIL Image (H_orig, W_orig, RGB)        ← original size (variable)
   ↓ np.array
numpy (H_orig, W_orig, 3) uint8        ← [0, 255]
   ↓ Image.fromarray + resize((256, 256))
PIL Image (256, 256, RGB)               ← standardized size
   ↓ F.to_image
torch.Tensor (3, 256, 256) uint8        ← channel first
   ↓ F.to_dtype(float32, scale=True)
torch.Tensor (3, 256, 256) float32       ← [0, 1]
   ↓ F.normalize(mean, std)
torch.Tensor (3, 256, 256) float32       ← mean ~ 0, variance ~ 1
   ↓ unsqueeze(0)
torch.Tensor (1, 3, 256, 256)            ← add batch dim
   ↓ model(tensor)
torch.Tensor (1, 8)                      ← 8 class logits
   ↓ softmax(dim=1)
torch.Tensor (1, 8)                      ← sums to 1.0
   ↓ [0]
torch.Tensor (8,)                        ← remove batch dim
   ↓ argmax()
int (e.g. 2)                              ← largest index
   ↓ classes[2]
str "Apple_rust"                          ← class name
```

**Once this whole flow is in your head, code reading becomes much easier.**

---

### F. Wrap-up — where each line connects

Most confusing points clarified:

1. **How LeafDataset applies augmentation fresh every time**:
   - `__getitem__` calls `self.transform(tensor)`.
   - `RandomHorizontalFlip`, `RandomRotation` are stateless — fresh random per call.
   - DataLoader calls `__getitem__` when building batches, so every step / every epoch sees different variants.

2. **What WeightedRandomSampler actually does**:
   - When DataLoader needs to pick batch_size=32 indices, it calls `sampler.__iter__()`.
   - Sampler picks indices weighted by `sample_weights` (replacement=True allows duplicates).
   - Result: batches contain small classes (Apple_rust) frequently, large classes (Apple_healthy) proportionally.

3. **Why train.py builds LeafDataset twice in `_build_loaders`**:
   - train_tf and val_tf differ. One dataset can't apply two transforms.
   - `discover_classes` sorts, so both datasets have identical sample order.
   - Therefore the indices from train_test_split work in both datasets.

4. **Why `metadata.json` is in the zip**:
   - Model weights alone don't tell you the class labels (0=Apple_Black_rot, etc).
   - `metadata.json` preserves the label mapping so predict.py can output class names.

5. **Why signature.txt is plain text, not zipped**:
   - On evaluation day, `make verify` needs to `diff` immediately against `shasum` output.
   - The standard `<sha1>  <basename>` format also works with `shasum -c signature.txt`.

---

## 🔗 Related Documents
- Design rationale: [docs/superpowers/specs/2026-04-28-leaffliction-design.md](superpowers/specs/2026-04-28-leaffliction-design.md)
- Implementation plan: [docs/superpowers/plans/2026-04-28-leaffliction.md](superpowers/plans/2026-04-28-leaffliction.md)
