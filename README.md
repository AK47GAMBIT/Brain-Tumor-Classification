# 🧠 Brain Tumor Classification with Transfer Learning (PyTorch)

A multi-class image classification project that detects and classifies **brain tumors from MRI scans** into 4 categories using **transfer learning with a pretrained ResNet18** backbone and a custom classifier head in PyTorch.

---

## 📌 Project Overview

This project builds an end-to-end deep learning pipeline that:

1. Loads and preprocesses MRI scan images with data augmentation
2. Visualizes a sample batch from the training set
3. Loads a pretrained **ResNet18** (ImageNet weights) as a frozen feature extractor
4. Replaces the original ResNet18 classifier with a custom 2-layer head for 4-class prediction
5. Trains for 20 epochs with validation tracked at every epoch
6. Saves the best-performing model checkpoint based on validation accuracy
7. Plots training and validation loss/accuracy curves

---

## 🗂️ Repository Structure

```
.
├── brain-tumor-classification-pytorch.ipynb   # Full pipeline: preprocessing, model, training, evaluation
├── best_model.pth                             # Best model weights (generated after training)
└── README.md
```

> **Note:** `best_model.pth` is not included in this repo. Run the notebook end-to-end to generate it — it is saved automatically whenever a new best validation accuracy is achieved during training.

---

## 📦 Dataset

- **Brain Tumor MRI Dataset** (Kaggle) — MRI scan images organized into `Training/` and `Testing/` folders
- **4 classes:** `glioma`, `meningioma`, `notumor`, `pituitary`
- Loaded using `torchvision.datasets.ImageFolder`

---

## 🧹 Data Preprocessing & Augmentation

All images pass through the same `transforms.Compose` pipeline:

- **Resize** to `224 × 224` (required by ResNet18)
- **RandomHorizontalFlip** — augmentation during training
- **RandomRotation** — up to ±10 degrees
- **ToTensor** — converts PIL images to `[0, 1]` float tensors
- **Normalize** — using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

DataLoaders use `batch_size=16`, with shuffling enabled for training only.

---

## 🧠 Model Architecture

`TumorClassifier` wraps a **pretrained ResNet18** as a frozen feature extractor with a custom classifier head:

```
TumorClassifier(
  (pretrained_model): ResNet18 (ImageNet weights, all layers frozen)
      Conv2d(3→64, 7×7) → BN → ReLU → MaxPool
      Layer1: 2 × BasicBlock (64 channels)
      Layer2: 2 × BasicBlock (128 channels)
      Layer3: 2 × BasicBlock (256 channels)
      Layer4: 2 × BasicBlock (512 channels)
      AdaptiveAvgPool2d → fc: Identity()   ← original head removed
  (classifier): Sequential(
      Linear(512 → 128)  →  ReLU
      Linear(128 → 4)
  )
)
```

- The ResNet18 backbone is **fully frozen** (`requires_grad=False`) — only the custom classifier head is trained
- The original `fc` layer is replaced with `nn.Identity()` so ResNet outputs raw 512-dim feature vectors
- Those features are passed into the 2-layer custom head for 4-class prediction
- **Loss:** `CrossEntropyLoss`
- **Optimizer:** `Adam` (`lr=0.001`)
- **Device:** CUDA if available, else CPU

---

## 📈 Training Results

Trained for **20 epochs** — best model checkpoint saved at epoch 20:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|---|---|---|---|
| 1 | 78.13% | 81.46% | 0.4748 |
| 5 | 87.62% | 84.59% | 0.3671 |
| 10 | 89.69% | 89.09% | 0.2898 |
| 15 | 90.67% | 90.01% | 0.2553 |
| 19 | 91.11% | 90.85% | 0.2289 |
| **20** | **92.07%** | **92.37%** | **0.1980** ← best |

- **Best Validation Accuracy: 92.37%** (Epoch 20)
- Validation loss steadily decreased across all 20 epochs, suggesting the model had not yet plateaued by the end of training

---

## 🚀 Usage

1. Download the **Brain Tumor MRI Dataset** from Kaggle and place `Training/` and `Testing/` folders in your working directory (or update the paths in the notebook)
2. Install dependencies (see below)
3. Run the notebook end-to-end — `best_model.pth` will be saved automatically

To load the saved model for inference:

```python
import torch
from torchvision import transforms
from PIL import Image

model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open("mri_scan.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Predicted:", classes[predicted_class])
```

---

## 📦 Requirements

```
torch
torchvision
numpy
matplotlib
```

Install with:
```bash
pip install torch torchvision numpy matplotlib
```

---
