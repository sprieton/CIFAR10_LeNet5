# 🔬 CIFAR10 LeNet5 with Temperature Scaling Analysis

**Deep Learning UC3M 2025 - Project 1: Understanding Calibration in CNNs**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 👥 **Team Members**
- *Álex Sánchez Zurita*
- *Jorge Barcenilla Gonzalez*
- *Santiago Prieto Núñez*

## 📋 **Project Overview**

This project investigates **model calibration** in Convolutional Neural Networks (CNNs) using a modernized LeNet5 architecture for binary classification of birds vs cats from CIFAR-10. The main focus is on implementing and analyzing **Temperature Scaling (Platt's Scaling)** to improve model calibration.

### 🎯 **Key Objectives**
1. **Train a modern LeNet5** from scratch for binary classification (birds vs cats)
2. **Evaluate model calibration** using reliability diagrams and Expected Calibration Error (ECE)
3. **Implement Temperature Scaling** to improve calibration without affecting accuracy
4. **Analyze the effect** of temperature parameter 'a' on model confidence
5. **Compare performance** before and after calibration using comprehensive metrics

## 🏗️ **Architecture & Improvements**

### **Modern LeNet5 Enhancements**
Our implementation modernizes the original LeNet5 with:

- ✅ **MaxPooling layers** instead of trainable subsampling
- ✅ **Batch Normalization** for training stability  
- ✅ **Increased filter counts** (32→64→128 vs original 6→16→120)
- ✅ **Dropout regularization** (p=0.5) to prevent overfitting
- ✅ **ReLU activations** for better gradient flow
- ✅ **Kaiming initialization** appropriate for ReLU networks
- ✅ **Adam optimizer** with learning rate scheduling

### **Architecture Details**
```
Input: 3×32×32 (RGB CIFAR-10 images)
├── Conv1: 32 filters, 5×5 → BatchNorm → ReLU → MaxPool(2×2)
├── Conv2: 64 filters, 5×5 → BatchNorm → ReLU → MaxPool(2×2)  
├── Conv3: 128 filters, 3×3 → BatchNorm → ReLU → MaxPool(2×2)
├── Flatten → Dropout(0.5)
├── FC1: 2048→512 → ReLU → Dropout(0.5)
├── FC2: 512→128 → ReLU
└── Output: 128→2 (Binary classification)
```

## 📊 **Dataset**

### **CIFAR-10 Subset**
- **Classes**: Birds (Class 2) → Label 0, Cats (Class 3) → Label 1
- **Training Set**: 10,000 images (5,000 per class)
- **Test Set**: 2,000 images (1,000 per class)
- **Preprocessing**: Normalization to [-1, 1] range

### **Data Download**
```bash
python dataset_downloader.py
```
This script automatically downloads CIFAR-10 and extracts only bird and cat images into organized folders.

## 🌡️ **Temperature Scaling Implementation**

### **What is Temperature Scaling?**
Temperature scaling is a post-processing calibration technique that improves the reliability of model confidence scores without changing predictions.

**Mathematical Formula:**
```
Original: p(y|x) = softmax(z_i)
Scaled:   p(y|x) = softmax(z_i / T)
```
Where `T` is the temperature parameter:
- `T > 1`: Makes model less confident (reduces overconfidence)
- `T < 1`: Makes model more confident  
- `T = 1`: No change (original model)

### **Calibration Metrics**
- **Expected Calibration Error (ECE)**: Measures the difference between confidence and accuracy
- **Reliability Diagrams**: Visual representation of calibration quality
- **Brier Score**: Combines calibration and sharpness metrics

## 📈 **Results & Performance**

### **Training Performance**
- **Final Training Accuracy**: ~96.86%
- **Test Accuracy**: ~87.15%
- **Model Parameters**: 1,243,010 trainable parameters
- **Training Epochs**: 35 with Adam optimizer and ReduceLROnPlateau scheduler

### **Calibration Results**
| Metric | Original Model | Temperature Scaled | Improvement |
|--------|---------------|-------------------|-------------|
| ECE | ~0.094 | ~0.015 | 84% reduction |
| Accuracy | 87.15% | 87.15% | No change ✅ |
| Calibration | Poor | Excellent | ✅ |

### **Key Findings**
- 🎯 **Optimal Temperature**: ~0.99 (model was already well-calibrated)
- 📉 **Significant ECE reduction** without accuracy loss
- 🔍 **Better reliability**: Confidence scores now match actual accuracy
- ⚖️ **Maintained performance**: No degradation in classification accuracy

## 🚀 **Getting Started**

### **Prerequisites**
```bash
pip install torch torchvision matplotlib numpy scikit-learn pandas seaborn scipy
```

### **Project Structure**
```
CIFAR10_LeNet5/
├── CIFAR10_LeNet5_P1.ipynb    # Main notebook with complete analysis
├── dataset_downloader.py       # CIFAR-10 data preparation
├── data/                       # Dataset folder (created automatically)
│   ├── train/bird/
│   ├── train/cat/
│   ├── test/bird/
│   └── test/cat/
├── README.md                   # This file
└── LICENSE
```

### **Running the Analysis**

1. **Download and prepare data:**
   ```bash
   python dataset_downloader.py
   ```

2. **Run the complete analysis:**
   Open `CIFAR10_LeNet5_P1.ipynb` in Jupyter Notebook or VS Code and run all cells.

3. **Key sections in the notebook:**
   - 📊 Dataset analysis and visualization
   - 🏗️ Model architecture definition  
   - 🎓 Training with learning rate scheduling
   - 🌡️ Temperature scaling implementation
   - 📈 Calibration analysis and visualization
   - 🔍 Confusion matrix and detailed metrics

## 📊 **Visualizations Included**

- **Dataset Distribution**: Class balance visualization
- **Training Curves**: Loss and accuracy over epochs with learning rate schedule
- **Reliability Diagrams**: Before and after temperature scaling
- **Calibration Analysis**: ECE vs temperature parameter effects
- **Confidence Distributions**: Original vs scaled model comparisons
- **Confusion Matrices**: Detailed classification performance analysis

## 🔬 **Technical Contributions**

1. **Modern CNN Architecture**: Updated LeNet5 with contemporary techniques
2. **Advanced Training**: Learning rate scheduling with ReduceLROnPlateau
3. **Calibration Analysis**: Comprehensive temperature scaling study
4. **Visualization Suite**: Multiple plots for understanding model behavior
5. **Performance Metrics**: ECE, Brier score, reliability diagrams, confusion matrices

## 📚 **Key References**

- **Paper**: ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) - Guo et al.
- **Original LeNet**: ["Gradient-Based Learning Applied to Document Recognition"](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) - LeCun et al.
- **CIFAR-10**: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html) - Krizhevsky

## 🎓 **Educational Value**

This project demonstrates:
- **Model Calibration**: Why confidence matters in AI systems
- **Temperature Scaling**: Simple yet effective calibration technique  
- **Modern CNN Design**: Best practices for neural network architecture
- **Comprehensive Evaluation**: Beyond accuracy - reliability and calibration
- **Real-world Applications**: Medical AI, autonomous systems, finance

## 🚨 **Important Notes**

- **Calibration vs Accuracy**: Temperature scaling improves confidence reliability without changing predictions
- **Post-processing**: Applied after training, doesn't require model retraining
- **Validation Split**: Uses 80% of test set for calibration, 20% for final evaluation
- **Reproducibility**: All random seeds and parameters documented

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---
**🏫 Course**: Deep Learning - Universidad Carlos III de Madrid (UC3M)  

