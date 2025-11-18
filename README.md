# Histopathologic Cancer Detection

A deep learning project for detecting metastatic cancer in histopathologic scans using transfer learning with ResNet50. This project was developed as part of Week 3 CNN assignment and participated in the [Histopathologic Cancer Detection Kaggle competition](https://www.kaggle.com/c/histopathologic-cancer-detection).

## 📋 Problem Description

The "Histopathologic Cancer Detection" challenge is a **binary image classification** problem. The goal is to develop a machine learning model that can automatically identify the presence of **metastatic cancer** in small image patches (96×96 pixels) derived from larger digital pathology scans of lymph node sections.

**Evaluation Metric:** Area Under the ROC Curve (AUC)

## 📊 Dataset

The dataset is a modified version of the **PatchCamelyon (PCam)** benchmark dataset:

- **Total Images:** 277,485 training images
- **Dataset Size:** 7.76 GB
- **Image Format:** `.tif` (Tagged Image File Format)
- **Patch Size:** 96 × 96 pixels (RGB)
- **Class Distribution:**
  - Non-Cancer (Label 0): 130,908 samples
  - Cancer (Label 1): 89,117 samples
  - Slight class imbalance present

The dataset is available on [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data).

## 🎯 Approach

This project employs a **two-phase transfer learning strategy** using ResNet50:

### Phase 1: Frozen Base Training
- **Base Model:** ResNet50 with ImageNet pre-trained weights (frozen)
- **Learning Rate:** 1.0 × 10⁻³
- **Epochs:** 8 (with early stopping)
- **Objective:** Train only the custom classification head while leveraging pre-trained features
- **Result:** Validation AUC = 0.9429

### Phase 2: Fine-Tuning
- **Base Model:** ResNet50 (unfrozen, fully trainable)
- **Learning Rate:** 1.0 × 10⁻⁵ (optimized via hyperparameter tuning)
- **Epochs:** 10 (with early stopping)
- **Objective:** Adapt pre-trained features to histopathology domain
- **Result:** Validation AUC = 0.9871

### Key Techniques
- **Transfer Learning:** ImageNet pre-trained ResNet50
- **Data Augmentation:** Rotation, shifts, flips, zoom
- **Class Weight Balancing:** Handles class imbalance
- **Hyperparameter Tuning:** Learning rate optimization on 10% subset
- **Early Stopping & Model Checkpointing:** Prevents overfitting

## 📈 Results

### Performance Summary

| Phase | Configuration | Validation AUC | Validation Loss | Accuracy |
|:------|:--------------|:--------------:|:--------------:|:--------:|
| **Phase 1** | Frozen Base, LR = 1e-3 | **0.9429** | 0.3162 | - |
| **Phase 2** | Fine-Tuned, LR = 1e-5 | **0.9871** | 0.1559 | 0.9461 |
| **Kaggle** | Final Model | **0.9236** (Private) / **0.9444** (Public) | - | - |

**Key Achievement:** Fine-tuning improved validation AUC by **+4.42 percentage points** (0.9429 → 0.9871).

### Hyperparameter Tuning Results

| Learning Rate | Validation AUC (10% Sample) | Selection |
|:--------------|:---------------------------:|:---------:|
| **1.0 × 10⁻⁵** | **0.9558** | ✅ Selected |
| 5.0 × 10⁻⁶ | 0.9420 | ❌ Too conservative |
| 1.0 × 10⁻⁶ | 0.9123 | ❌ Too slow convergence |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- GPU support (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/hoangop/Deep_Learning_W3_project.git
cd Deep_Learning_W3_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- TensorFlow 2.16.2
- Keras 3.12.0
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.10.7
- Scikit-learn 1.7.2
- Pillow 12.0.0
- Seaborn 0.13.2

## 📖 Usage

### Running the Notebook

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/data) and place it in the appropriate directory structure.

2. **Open the Jupyter notebook:**
```bash
jupyter notebook notebook/PROJECT_11_17.ipynb
```

3. **Update data paths** in the notebook to match your local setup:
```python
INPUT_DIR = '/path/to/histopathologic-cancer-detection'
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')
```

4. **Run cells sequentially** to:
   - Perform exploratory data analysis
   - Train Phase 1 (frozen base)
   - Perform hyperparameter tuning
   - Train Phase 2 (fine-tuning)
   - Generate predictions and submission file

### Training Process

The notebook is organized into the following sections:

1. **Data Loading & EDA:** Exploratory data analysis and visualization
2. **Model Architecture:** ResNet50 with custom classification head
3. **Phase 1 Training:** Frozen base training
4. **Hyperparameter Tuning:** Learning rate optimization
5. **Phase 2 Training:** Fine-tuning on full dataset
6. **Results & Analysis:** Comprehensive performance analysis
7. **Submission:** Generate predictions for Kaggle submission

## 📁 Project Structure

```
Deep_Learning_W3_project/
│
├── notebook/
│   ├── PROJECT_11_17.ipynb          # Main project notebook
│   └── best_model_resnet_final.weights.h5  # Trained model weights
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🔍 Key Findings

### What Worked Well
- **Transfer Learning:** Enabled immediate high performance (AUC > 0.94) without training from scratch
- **Two-Phase Strategy:** Freezing base layers first, then fine-tuning, provided stable training
- **Hyperparameter Tuning on Subset:** Testing on 10% data saved GPU time while providing reliable guidance
- **Data Augmentation:** Essential for preventing overfitting on small patches

### Challenges & Lessons Learned
- **Training Time Constraints:** Deep learning models require extensive training time, which limited the number of hyperparameter tuning experiments
- **Computational Efficiency:** Better module structuring and code organization would optimize training time and enable more comprehensive experimentation
- **Learning Rate Selection:** 100× reduction (1e-3 → 1e-5) was critical for fine-tuning to prevent catastrophic forgetting

## 📚 References

- **Competition:** [Histopathologic Cancer Detection on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)
- **Dataset:** PatchCamelyon (PCam) benchmark dataset
- **Model Architecture:** ResNet50 (He et al., 2016)
- **Framework:** TensorFlow/Keras

## 👤 Author

**Phương Hoàng Nguyễn**

## 📝 License

This project is for educational purposes as part of a course assignment.

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- TensorFlow/Keras team for the deep learning framework
- The PatchCamelyon dataset creators

---

**Note:** This project was developed as part of a Deep Learning course assignment. The model achieved competitive performance on the Kaggle leaderboard, demonstrating the effectiveness of transfer learning for medical image classification tasks.
