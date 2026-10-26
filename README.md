# Frozen vs. Fine-Tuned ResNet50

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=google-colab&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)

Controlled comparison of linear probing (frozen backbone) versus two-stage fine-tuning on a 30-class, fine-grained image dataset.

## Results

All results are averaged over **5 random seeds** (`[0, 1, 2, 3, 4]`) using a fixed, stratified 80/10/10 train/val/test split (`SPLIT_SEED=42`). The test set was evaluated exactly once per seed using the best-validation checkpoint.

| Condition | Val Accuracy | Test Accuracy |
| :--- | :--- | :--- |
| **Frozen Backbone** | 81.57% ± 0.53% | 80.29% ± 1.03% |
| **Fine-Tuned** | **95.71% ± 0.45%** | **93.43% ± 1.03%** |

**Delta:** Fine-tuning the final convolutional block (`layer4`) alongside the linear head yields a **+13.1 percentage point** improvement on the held-out test set. Notably, the test variance (±1.03%) is identical across both conditions, indicating that the improvement is driven by model capacity, not split-level noise.

## The Dataset & Data Quality

The dataset consists of 30 classes of characters from the video game *Genshin Impact*.
To my knowledge, there was no existing standard dataset for this 30-class task. I hand-collected ~1,400 images and **intentionally hand-cropped and processed the dataset manually** instead of writing a script. I wanted to experience the raw, unscripted reality of real-world data collection for my first major dataset.

This manual approach resulted in a dataset with real-world noise:
- Inconsistent aspect ratios and cut-off heads (e.g., zooming in on a character's face).
- Multi-character images where a secondary character occupies a large portion of the frame.
- Slight label noise due to the ambiguity of multi-character official art.

I kept the noise because it simulates what a model actually faces in deployment.

### Failure Analysis: Visual Similarity vs. Label Noise
The fine-tuned model achieved 93.43% test accuracy, but the confusion matrices (see `results/finetuned/`) revealed three distinct types of systematic errors:

1. **Visual Similarity:** Across multiple seeds, the model frequently confused specific blue-themed classes. Visual inspection of the misclassified samples reveals these characters share cool-toned color palettes, specific headwear silhouettes, and similar framing. The model is relying on low-level visual features rather than semantic identity.
2. **Label Noise:** The model consistently struggled to separate two specific classes. Upon reviewing the training data, I found the root cause: several images labeled as Class A were actually dual-character images where Class B occupied roughly 40% of the frame. The model was not failing; it was accurately learning from a noisy label and associating Class B's features with the Class A label.

## Methodology

- **Architecture:** ResNet50 with `IMAGENET1K_V2` weights.
- **Frozen Baseline:** 30 epochs, backbone frozen, AdamW optimizer (`lr=1e-3`), standard CrossEntropy.
- **Fine-Tuning:** 40 epochs, two-stage training. 
  - Epochs 0–2: Head-only training.
  - Epochs 3–39: Unfreeze `layer4`. Discriminative learning rates applied (Head `1e-3`, Backbone `1e-4`) with Cosine Annealing. 
  - Uses Automatic Mixed Precision (AMP), Label Smoothing (`0.1`), and augmentation (`RandomResizedCrop`, `ColorJitter`, `RandomErasing`).
- **Evaluation:** No early stopping. The model trains for the full schedule, ensuring the cosine LR reaches ~0. The best-validation checkpoint is saved and used for the final test evaluation.

## Reproducibility

Tested on Google Colab (Python 3.13.16, PyTorch 2.11.0, CUDA 13.0).

**Notebooks:**
- `frozen30.ipynb`: Trains the baseline across 5 seeds.
- `finetune30.ipynb`: Trains the fine-tuned pipeline across 5 seeds.

**Local Setup:**
The environment is managed with `uv`. To sync dependencies locally:
```bash
uv sync
```

All training curves, confusion matrices, and raw JSON metrics are saved in the results/ directory. Model weights (.pth) and the raw image dataset are not committed to this repository to keep it lightweight.

## Limitations
**Dataset Size:** With ~1,400 images (~46 per class), the model is highly prone to overfitting. The heavy augmentation and label smoothing in the fine-tuned pipeline were necessary to stabilize training.

**Single Split:** While 5 seeds are used for model initialization, the train/val/test split is held fixed. Performance may vary slightly on a different random split of the images.

**Domain Shift:** The dataset consists of stylized 2D anime art. Transferring ImageNet features (real-world photos) to this domain requires aggressive fine-tuning of the high-level features (layer4), which is why the frozen baseline plateaus at 80%.