# Plant Leaf Image Super-Resolution using SRGAN

A deep learning project for **4× single-image super-resolution** of degraded plant-leaf images.

The task is to reconstruct a high-resolution `128 × 128` RGB image from a severely degraded `32 × 32` RGB input. The project investigates reconstruction losses, residual networks, conditional adversarial training, and pixel-shuffle based upsampling. It also includes experiments with **VGG-19 perceptual loss** and alternative **Charbonnier loss** formulations to balance pixel accuracy against realistic texture.

**Best leaderboard result:** Private MAE **16.7747859** · Public MAE **16.1249961** (lower is better).

## 1. Problem

The problem is a 4× super-resolution task:

```text
Low Resolution Image
32 × 32 × 3
       │
       ▼
   Deep Network
       │
       ▼
High Resolution Image
128 × 128 × 3
```

The low-resolution images contain substantial information loss and compression/noise artifacts. The model therefore has to learn both:

1. **Spatial reconstruction** — increasing the resolution by 4×.
2. **Texture reconstruction** — recovering high-frequency structures that are weak or absent in the low-resolution input.

The target images are pristine `128 × 128` RGB images.

---

## 2. Dataset

The dataset contains:

| Split | Images | Resolution |
|---|---:|---|
| Training LR | 1,642 | 32 × 32 × 3 |
| Training HR | 1,642 | 128 × 128 × 3 |
| Test LR | 495 | 32 × 32 × 3 |

Each low-resolution training image has a corresponding high-resolution image with the same filename.

The test set contains only the low-resolution images. The model must generate the corresponding high-resolution reconstruction.

A pretrained VGG-19 weight file was also provided by the task. I used it for **perceptual-loss experiments** (Section 7.4), where VGG-19 serves as a fixed feature extractor. The generator and discriminator themselves are trained from scratch.

---

## 3. Approach

The project evolved around a **SRGAN/SRResNet-style reconstruction pipeline**.

The main idea was to first learn reliable image reconstruction and then introduce adversarial learning rather than allowing the GAN objective to dominate from the beginning.

The final training strategy can be summarized as:

```text
32×32 LR Image
      │
      ▼
  Generator
      │
      ├── Initial convolution
      ├── 16 residual blocks
      ├── Residual feature fusion
      ├── PixelShuffle ×2
      └── PixelShuffle ×2
      │
      ▼
128×128 Super-Resolved Image
      │
      ├───────────────┐
      ▼               ▼
Reconstruction     Discriminator
Loss                    │
      │                 │
      └───────┬─────────┘
              ▼
        Generator Update
```

The discriminator is **conditional**: it receives both the generated/high-resolution image and the corresponding upsampled low-resolution input.

---

## 4. Generator

The generator follows a residual super-resolution architecture.

### Initial feature extraction

The input is converted from 3 RGB channels into 64 feature channels:

```python
nn.Conv2d(3, 64, 3, padding=1)
nn.PReLU()
```

This produces a feature representation while preserving the `32 × 32` spatial resolution.

---

### Residual trunk

The network contains **16 residual blocks**.

Each block contains:

```text
3×3 Convolution
      ↓
Batch Normalization
      ↓
PReLU
      ↓
3×3 Convolution
      ↓
Batch Normalization
      ↓
Residual Addition
```

Mathematically, a residual block learns a transformation $F(x)$ and returns:

```math
y = x + F(x)
```

Instead of forcing the network to learn an entirely new representation, the block only needs to learn the useful residual correction.

This is particularly useful for super-resolution because much of the low-frequency image structure should remain consistent while the network focuses on reconstructing missing details.

---

### Global residual connection

After the residual stack, an additional convolution and batch-normalization layer are combined with the features produced at the generator entrance:

```math
r = \text{Mid}(R(f)) + f
```

where:

- $f$ is the initial feature representation;
- $R$ represents the residual trunk;
- $r$ is the fused feature representation.

This gives the network a direct path for preserving information from the original low-resolution representation.

---

## 5. Upsampling

The generator performs the 4× resolution increase using **PixelShuffle**.

The upsampling sequence is:

```text
32 × 32
   │
PixelShuffle(2)
   ↓
64 × 64
   │
PixelShuffle(2)
   ↓
128 × 128
```

Before each PixelShuffle operation, the number of feature channels is increased from 64 to 256:

```python
nn.Conv2d(64, 256, 3, padding=1)
nn.PixelShuffle(2)
```

PixelShuffle rearranges channel information into spatial dimensions.

For an upscale factor $r$, PixelShuffle transforms:

```math
(Cr^2, H, W)
```

into:

```math
(C, Hr, Wr)
```

This provides a learned alternative to simply interpolating the image.

---

## 6. Discriminator

The discriminator is conditional.

Instead of looking only at the generated image, it receives:

```text
Super-Resolved Image
        +
Upsampled Low-Resolution Image
        ↓
Concatenation
        ↓
6-channel input
```

The upsampled LR image is created using bicubic interpolation:

```python
lr_up = F.interpolate(
    lr,
    size=(128, 128),
    mode="bicubic"
)
```

The discriminator therefore evaluates the generated HR image in the context of the original LR observation.

Its convolutional feature extractor increases the number of channels:

```text
6
↓
64
↓
128
↓
256
↓
512
```

followed by global average pooling and a single scalar output.

---

## 7. Reconstruction Loss

A major part of the training approach was to make the generator learn faithful reconstruction before relying heavily on adversarial training.

The reconstruction objective is:

```math
L_{\text{pix}}
=
L_{\text{Charbonnier}}
+
0.5L_1
+
0.1L_{\text{gradient}}
```

---

### 7.1 Charbonnier Loss

The implementation uses:

```math
L_{\text{Charbonnier}}
=
\frac{1}{N}
\sum_i
\sqrt{(x_i-y_i)^2+\epsilon}
```

with:

```math
\epsilon = 10^{-6}
```

The Charbonnier loss is a smooth approximation to the absolute error.

For small errors, it behaves smoothly around zero, while for larger errors it remains similar to an $L_1$ objective.

---

### 7.2 L1 Reconstruction Loss

The second component is standard mean absolute error:

```math
L_1
=
\frac{1}{N}
\sum_i |x_i-y_i|
```

where $x_i$ is the generated pixel and $y_i$ is the ground-truth pixel.

The motivation is straightforward: super-resolution should remain numerically faithful to the ground-truth image.

---

### 7.3 Gradient Loss

The model also compares image gradients using Sobel filters.

The horizontal and vertical kernels are:

```text
[-1  0  1]       [-1 -2 -1]
[-2  0  2]       [ 0  0  0]
[-1  0  1]       [ 1  2  1]
```

For each RGB channel, the generated and target gradients are compared using L1 loss.

Conceptually:

```math
L_{\text{gradient}}
=
\left\|
\nabla_x I_{SR}-\nabla_x I_{HR}
\right\|_1
+
\left\|
\nabla_y I_{SR}-\nabla_y I_{HR}
\right\|_1
```

The purpose is to encourage preservation of edges and high-frequency structure.

This is important for leaf images because veins and fine biological structures are represented primarily through local intensity changes.

---

### 7.4 Perceptual Loss (VGG-19) — Experiment

Pixel-wise losses such as L1 and Charbonnier average over every plausible high-resolution image, which tends to produce smooth, blurry textures.

A perceptual loss instead compares images in the **feature space of a pretrained VGG-19 network**, where edges, textures and patterns are represented explicitly:

```math
L_{\text{perceptual}}
=
\left\|
\phi(I_{SR})-\phi(I_{HR})
\right\|
```

where $\phi$ represents features extracted by VGG-19.

Adding this term to the generator objective pushes the output toward realistic texture, such as leaf veins and surface detail, rather than toward a pixel-wise average.

---

### 7.5 Charbonnier Variants — Experiment

I also experimented with alternative formulations of the Charbonnier term to study how the smooth L1-style reconstruction loss interacts with the L1, gradient, perceptual and adversarial components.

---

### 7.6 What the Loss Experiments Showed

The leaderboard metric is MAE, which directly rewards pixel accuracy.

Perceptual and adversarial terms improve how realistic the texture looks, but they can move pixel values away from the ground truth. Their weights therefore had to stay small relative to the reconstruction objective.

All variants landed in a narrow range (private MAE **16.77–16.80**). The best experimental submission reached a private MAE of **16.7747859**. See Section 15 for all submissions.

---

## 8. Adversarial Training

The discriminator uses binary cross entropy with logits:

```math
L_D
=
\frac{1}{2}
\left[
\text{BCE}(D(HR),1)
+
\text{BCE}(D(SR),0)
\right]
```

The generator receives an adversarial objective:

```math
L_{adv}
=
\text{BCE}(D(SR),1)
```

The generator attempts to produce images that the discriminator considers realistic.

However, directly optimizing a GAN from the beginning can make reconstruction unstable. Therefore the training schedule intentionally separates reconstruction learning from adversarial learning.

---

## 9. Two-Stage GAN Training

The final training loop uses **90 epochs**.

### Stage 1 — Reconstruction

For the first 60 epochs:

```python
g_loss = pix
```

The generator is trained only using:

```text
Charbonnier
+
L1
+
Gradient loss
```

This allows it to first learn the deterministic LR → HR mapping.

### Stage 2 — Adversarial refinement

From epoch 60 onward:

```python
g_loss = pix + 0.001 * adv
```

The adversarial contribution is intentionally small.

Thus:

```math
L_G =
L_{\text{pix}}
+
0.001L_{\text{adv}}
```

The reconstruction objective remains dominant while the discriminator provides an additional realism signal.

---

## 10. Stabilization Choices

Several choices were made to reduce GAN instability.

### Different learning rates

The generator uses:

```text
2 × 10⁻⁴
```

while the discriminator uses:

```text
1 × 10⁻⁵
```

The discriminator therefore learns much more slowly than the generator.

This was a deliberate attempt to prevent the discriminator from becoming too strong too quickly.

### Label smoothing

The real discriminator target is changed from exactly `1` to:

```text
0.9
```

while the fake target is changed from exactly `0` to:

```text
0.1
```

This prevents the discriminator from receiving perfectly confident binary targets.

### Generator output constraint

Generated images are clipped to the valid normalized image range:

```python
sr = torch.clamp(sr, 0, 1)
```

before loss computation and output conversion.

---

## 11. Data Augmentation

The paired LR and HR images receive the same spatial transformations.

Two independent augmentations are used:

```text
Horizontal Flip
Vertical Flip
```

Each is applied with probability 0.5.

The important property is that the transformation is applied identically to both images.

For example:

```text
LR:  ──flip──> LR'
HR:  ──flip──> HR'
```

This preserves the pixel correspondence between the input and target.

---

## 12. Training Configuration

| Parameter | Value |
|---|---:|
| Input size | 32 × 32 |
| Target size | 128 × 128 |
| Batch size | 8 |
| Epochs | 90 |
| Generator LR | 2e-4 |
| Discriminator LR | 1e-5 |
| Optimizer | Adam |
| Residual blocks | 16 |
| Feature channels | 64 |
| Upsampling | PixelShuffle ×2 ×2 |
| GAN activation | Epoch 60 onward |
| Adversarial weight | 0.001 |

---

## 13. Evaluation

The leaderboard metric is **Mean Absolute Error (MAE)**.

For predicted pixels $\hat{y}$ and target pixels $y$:

```math
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
```

Lower values indicate better reconstruction.

The leaderboard evaluates the generated `128 × 128 × 3` images against the hidden high-resolution targets.

---

### Important evaluation note

The notebook's `eval_loader` is constructed from the same paired training dataset:

```python
eval_loader = DataLoader(
    SRDataset(lr_path, hr_path, False),
    batch_size=8,
    shuffle=False
)
```

Therefore, the MAE printed during training is **not a held-out validation score**. It measures reconstruction on the available training pairs.

The actual generalization result is the leaderboard score obtained from the hidden test targets.

---

## 14. Submission

The final generated image is converted back from normalized floating-point values to 8-bit RGB:

```python
sr = (sr * 255).clip(0, 255).astype(np.uint8)
```

The `128 × 128 × 3` image is then flattened into a single sequence of:

```math
128 \times 128 \times 3 = 49,152
```

pixel values.

The submission format is:

```text
Id,Pixels
image_name.png,<49,152 space-separated pixel values>
```

---

## 15. Final Leaderboard Result

| Submission | Private MAE | Public MAE |
|---|---:|---:|
| **Best experimental variant (loss experiments)** | **16.7747859** | **16.1249961** |
| Notebook in this repository (Charbonnier + L1 + gradient + delayed GAN) | 16.7806549 | 16.1389827 |
| Other experimental variant | 16.7954559 | 16.1406822 |

Because the evaluation metric is MAE:

```math
\boxed{\text{Lower is better}}
```

The difference between public and private performance also illustrates why local reconstruction error alone is not sufficient for judging generalization.

---

## 16. What I Learned From the GAN Approach

This project was not simply an exercise in implementing a GAN architecture.

A significant part of the work was understanding the practical difficulty of adversarial super-resolution.

The central problem is that two objectives can conflict:

```text
Pixel reconstruction
        ↓
Stay faithful to ground truth

Adversarial objective
        ↓
Produce perceptually realistic details
```

A generator optimized only for pixel-wise error tends to produce conservative, smooth reconstructions.

A generator given too much adversarial or perceptual pressure can instead produce visually convincing details that are not actually present in the target.

For this reason, the final training strategy deliberately kept:

```math
L_{\text{pixel}}
\gg
L_{\text{adversarial}}
```

through the very small adversarial coefficient:

```math
\lambda_{adv}=0.001
```

and delayed adversarial training until after the reconstruction stage.

This experimentation was an important part of the project because it exposed the difference between **image reconstruction** and **image generation**.

---

## 17. Technical Takeaways

### 1. Super-resolution is not simply resizing

Bicubic interpolation can increase the image dimensions, but it cannot recover information that has been lost.

The neural network instead learns a mapping:

```math
f_\theta:
I_{LR}\rightarrow I_{HR}
```

from 1,642 paired training examples.

### 2. Residual learning helps deep reconstruction

Residual blocks make it easier for the network to learn corrections instead of rebuilding the entire representation.

### 3. PixelShuffle provides learned upsampling

Rather than interpolating feature maps directly, the network learns how feature channels should be rearranged into higher-resolution spatial information.

### 4. Gradient information matters

Pixel losses alone can encourage smooth outputs. Comparing image gradients explicitly encourages the network to preserve edges and fine structures.

### 5. Perceptual loss trades pixel accuracy for realism

VGG-19 feature matching produces sharper, more natural textures, but under an MAE metric its weight has to be controlled carefully.

### 6. GAN training is a balancing problem

The discriminator is not automatically beneficial simply because it is present.

The relative strength of:

```math
L_{\text{reconstruction}}
\quad\text{and}\quad
L_{\text{adversarial}}
```

strongly affects the behaviour of the generator.

---

## 18. Reproducibility

The notebook contains the complete implementation used for the experiment:

```text
Data loading
      ↓
Paired augmentation
      ↓
Generator
      ↓
Discriminator
      ↓
Combined reconstruction loss
      ↓
Two-stage GAN training
      ↓
MAE evaluation
      ↓
Test inference
      ↓
CSV submission
```

The notebook contains the Charbonnier + L1 + gradient + delayed-adversarial configuration. The perceptual-loss and Charbonnier-variant experiments were separate runs.

The repository does not include the original dataset or generated submission data.

---

## 19. Repository Structure

The project intentionally contains only the main documentation and experiment notebook:

```text
.
├── README.md
└── Plant_Resolution_Experiment.ipynb
```

The notebook contains the implementation and experimental workflow.

The dataset remains external and is not committed to the repository.

---

## 20. Future Work

Several improvements would be natural extensions of this experiment:

### Systematic perceptual-loss study

Compare different VGG-19 feature layers and perceptual-loss weights, and report the texture-versus-MAE trade-off for each.

### Stronger reconstruction architecture

Possible directions include:

```text
RRDB
EDSR
RCAN
SwinIR
```

### Improved GAN objective

The discriminator could be replaced or extended with:

```text
Hinge loss
Relativistic GAN
WGAN-GP
```

### Better validation methodology

A dedicated held-out validation split would allow model selection and hyperparameter tuning without relying on the hidden leaderboard.

### Controlled ablations

Future experiments could separately measure the contribution of:

```text
L1
Charbonnier
Gradient loss
Perceptual loss
Adversarial loss
Residual depth
PixelShuffle
Label smoothing
```

This would make it possible to quantify which components actually improve reconstruction rather than changing several components simultaneously.

---

## 21. Final Result

The best submission achieved the following leaderboard results:

| Metric | Score |
|---|---:|
| **Private leaderboard MAE** | **16.7747859** |
| Public leaderboard MAE | 16.1249961 |

The evaluation metric is **Mean Absolute Error (MAE)**, so:

```text
Lower = Better
```

The private leaderboard score is the primary final result because it represents the final hidden evaluation.

---

## 22. Summary

The project explored a complete neural super-resolution pipeline for reconstructing `128 × 128` plant-leaf images from `32 × 32` degraded observations.

The final architecture combines:

```text
Residual Generator
        +
PixelShuffle Upsampling
        +
Conditional Discriminator
        +
Charbonnier Loss
        +
L1 Loss
        +
Gradient Loss
        +
Delayed Adversarial Training
        +
VGG-19 Perceptual Loss (experiments)
```

The best submitted result was:

```math
\boxed{\text{Private MAE}=16.7747859}
```

with a public leaderboard MAE of:

```math
\boxed{\text{Public MAE}=16.1249961}
```

The most important engineering lesson from the project was that successful GAN-based super-resolution requires balancing **faithful reconstruction** against **perceptual realism**. The adversarial component is useful only when it complements the reconstruction objective rather than overwhelming it.
