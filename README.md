# Few-Shot Medical Image Synthesis for Rare Skin-Lesion Classes
### Does adversarial and perceptual supervision make few-shot generated images more useful than a plain VAE?

**Data:** HAM10000 dermatoscopic images (research use only, not for clinical use)

---

**Research question.** When a skin-lesion class has only **5 training images**, can a few-shot generative model create useful extra images, and does adversarial + perceptual supervision make them more useful than a plain VAE?

## 1. Method

1. **Data.** HAM10000 (`kmader/skin-cancer-mnist-ham10000`, downloaded with `kagglehub`, no login), center-cropped and resized to 64×64. 10,015 images from 7,470 lesions.
2. **Lesion-level split** inside each class (70 % train, 10 % validation, 20 % test), so images of the same lesion never appear in both training and test sets. This avoids the leakage that is common with HAM10000.
3. **Few-shot setup.** The two rarest classes (**df** dermatofibroma and **vasc** vascular lesions) keep only **5 training images** each, taken from 5 different lesions. All other classes keep their full training data.
4. **Two few-shot generators** with the same architecture:
   - a support encoder embeds each support image; the class vector is the **mean** of these embeddings (as in my thesis FS-CVAE);
   - an encoder gives the latent code of a query image given the class vector;
   - a decoder generates a 64×64 image from the latent code and class vector;
   - training is episodic (4 support images, 1 target image from the same class).

   The two variants:
   - **FS-CVAE:** L1 reconstruction + KL.
   - **FS-CVAE + PatchGAN + perceptual:** adds a PatchGAN discriminator and a VGG16 perceptual loss.
5. **Quality of generated images**, judged by an independent ResNet-18 trained on all real training images: class fidelity, sharpness (Laplacian variance) and a copy ratio (distance to the 5 support images relative to real test images; much below 1 means copying).
6. **Downstream classifier** (ResNet-18, ImageNet weights, 15 epochs, 3 seeds) under five conditions:

| Condition | Training data |
|---|---|
| **A** | Only the 5 real images per rare class |
| **B** | A + heavy augmentation of the 5 images (strong baseline) |
| **C** | A + FS-CVAE images |
| **D** | A + FS-CVAE + PatchGAN + perceptual images |
| **E** | All real images (upper bound) |

**The fair comparison is D vs. B**: both add the same number of images (300 per rare class) built from the same 5 real images. **D vs. C** is the ablation of the adversarial + perceptual supervision. Paired bootstrap intervals are computed on the rare-class test images.

## 3. Setup actually used

| Item | Value |
|---|---|
| Image size | 64×64 |
| Rare classes | df (84 training images in full data), vasc (97) |
| Shots per rare class | 5 |
| Added images per rare class | 300 |
| Generator training steps | 6000 |
| Test images in rare classes | df 21, vasc 27 |
| Total test images | 2011 |

## 4. Results

**Generated image quality** (independent classifier):

| Class | Source | Fidelity % | Sharpness | Copy ratio |
|---|---|---|---|---|
| df | real test images | 47.6 | 0.003 | 1.00 |
| df | FS-CVAE | 0.0 | 0.000 | 0.83 |
| df | FS-CVAE + PatchGAN + perceptual | 66.7 | 0.001 | 0.84 |
| vasc | real test images | 88.9 | 0.003 | 1.00 |
| vasc | FS-CVAE | 30.7 | 0.000 | 0.79 |
| vasc | FS-CVAE + PatchGAN + perceptual | 72.0 | 0.001 | 0.79 |

The PatchGAN + perceptual model produces images that the independent classifier recognises as the intended class far more often (df 0 % to 66.7 %, vasc 30.7 % to 72.0 %) and that are sharper than the plain VAE, whose images are very blurry. Copy ratios around 0.8 mean the images lie somewhat closer to the 5 support images than real test images do, so some copying of the support set is present.

**Classification** (mean over 3 seeds):

| Condition | Rare-class recall % | Balanced acc. % | Macro F1 % | Recall df | Recall vasc |
|---|---|---|---|---|---|
| A: 5 rare images | 16.7 ± 1.9 | 46.9 | 49.6 | 0.0 | 33.3 |
| B: + heavy augmentation | 25.5 ± 2.4 | 50.8 | 53.3 | 1.6 | 49.4 |
| C: + FS-CVAE | 26.5 ± 1.1 | 50.6 | 52.7 | 0.0 | 53.1 |
| D: + FS-CVAE + PatchGAN + perceptual | 24.8 ± 1.6 | 49.2 | 52.2 | 6.3 | 43.2 |
| E: all real images (upper bound) | 66.8 ± 6.1 | 60.2 | 62.5 | 46.0 | 87.7 |

**Paired bootstrap on rare-class recall (95 % CI):**
- D minus B: −1.4 points (−6.9 to +4.2)
- D minus C: −2.8 points (−9.0 to +3.5)
- C minus B: +1.4 points (−2.8 to +5.6)

## 5. Interpretation

- Adding anything built from the 5 images (B, C or D) clearly helps over using the 5 images alone (A: 16.7 % to about 25 %).
- **Generated images did not beat plain heavy augmentation.** All three paired differences have confidence intervals that include zero, so in this run neither the VAE nor the adversarial + perceptual version gives a measurable benefit over the augmentation baseline.
- The PatchGAN + perceptual supervision improved *visual* quality and class fidelity but did not translate into better recall. Higher fidelity does not guarantee more useful training signal, especially when images stay close to the 5 support images.
- There is a large gap to the upper bound (66.8 %), showing how hard 5-shot learning is for these classes. The df class is almost never recognised under conditions A to C.

