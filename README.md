# Edge AI-Based Diabetic Retinopathy Detection Using CNN and FPGA Acceleration

This project implements a deep learning-based diabetic retinopathy detection system using retinal fundus images. A binary Convolutional Neural Network (CNN) is trained to classify images as either Diabetic Retinopathy (DR) or No Diabetic Retinopathy (No_DR). The project also includes Grad-CAM explainability and an FPGA-oriented HLS convolution accelerator to support edge AI deployment.

---

## Project Overview

Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early. This project focuses on automated retinal image classification using a CNN model and explores FPGA acceleration for low-power edge inference.

The system includes:

- Retinal image preprocessing
- Binary classification of DR and No_DR
- CNN model training using TensorFlow/Keras
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Grad-CAM visualization for explainability
- C++ HLS-based 3×3 convolution accelerator for FPGA deployment

---

## Dataset

The project uses retinal fundus images organized into five original classes:

- No_DR
- Mild
- Moderate
- Severe
- Proliferate_DR

For binary classification, the classes are mapped as follows:

| Original Class | Binary Class |
|---|---|
| No_DR | No_DR |
| Mild | DR |
| Moderate | DR |
| Severe | DR |
| Proliferate_DR | DR |

---

## Model Architecture

The CNN model consists of:

- Conv2D layer with 32 filters
- MaxPooling2D layer
- Conv2D layer with 64 filters
- MaxPooling2D layer
- Conv2D layer with 128 filters
- MaxPooling2D layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer
- Sigmoid output layer for binary classification

The model is trained using:

- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Image size: 224 × 224
- Batch size: 32
- Epochs: 20
- Best model selection using validation accuracy

---

## Evaluation Results

The best trained model achieved the following validation performance:

| Metric | Value |
|---|---:|
| Accuracy | 95% |
| DR Precision | 0.96 |
| DR Recall | 0.95 |
| DR F1-score | 0.95 |
| No_DR Precision | 0.95 |
| No_DR Recall | 0.96 |
| No_DR F1-score | 0.95 |

The validation set contained 732 images.

### Confusion Matrix

| True / Predicted | DR | No_DR |
|---|---:|---:|
| DR | 352 | 19 |
| No_DR | 15 | 346 |

This shows balanced classification performance across both DR and No_DR classes.

---

## Grad-CAM Explainability

Grad-CAM was implemented to visualize the retinal regions that influenced the CNN prediction. This improves interpretability by showing whether the model focuses on clinically relevant areas of the fundus image.

The Grad-CAM output is saved in:

```text
results/gradcam_output.png
```

---

## FPGA / HLS Acceleration

A 3×3 convolution accelerator was implemented in C++ for FPGA deployment using Vitis HLS. Since convolution is one of the most computationally intensive operations in CNN-based image processing, this module represents the hardware acceleration component of the system.

The accelerator accepts a 224 × 224 grayscale image and applies a 3 × 3 convolution kernel to generate a 222 × 222 output feature map.

### HLS Files

```text
hls/
├── conv2d_accelerator.cpp
├── conv2d_accelerator.h
└── testbench.cpp
```

### HLS C++ Simulation

A C++ testbench was created using an artificial edge image. The convolution output around the edge region was:

```text
0 -3 3 0 0 0 0 0
0 -3 3 0 0 0 0 0
0 -3 3 0 0 0 0 0
0 -3 3 0 0 0 0 0
0 -3 3 0 0 0 0 0
```

This confirms that the convolution accelerator successfully detects edge transitions in the input image.

---

## Project Structure

```text
dr_detection_project/
├── src/
│   ├── train_binary.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── preprocess.py
│
├── hls/
│   ├── conv2d_accelerator.cpp
│   ├── conv2d_accelerator.h
│   └── testbench.cpp
│
├── results/
│   ├── confusion_matrix.png
│   ├── evaluation_report.txt
│   └── gradcam_output.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Train the model

```powershell
python src/train_binary.py
```

### 4. Evaluate the model

```powershell
python src/evaluate.py
```

### 5. Run prediction on a single image

```powershell
python src/predict.py "path_to_image"
```

### 6. Run Grad-CAM

```powershell
python src/gradcam.py "path_to_image"
```

### 7. Compile and run HLS C++ simulation

```powershell
D:\C-C++\ucrt64\bin\g++.exe -fno-lto hls\testbench.cpp hls\conv2d_accelerator.cpp -o hls\conv_test.exe
.\hls\conv_test.exe
```

---

## Results Files

The generated result files are stored in the `results/` directory:

```text
results/
├── confusion_matrix.png
├── evaluation_report.txt
└── gradcam_output.png
```

---

## Future Work

- Perform full Vitis HLS synthesis
- Generate RTL/IP from the convolution accelerator
- Integrate the accelerator with a Zynq-based FPGA design
- Deploy inference on PYNQ-Z2 or Zybo Z7-20
- Compare CPU inference time with FPGA-accelerated inference
- Extend the model to multi-class DR severity classification

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- C++
- Vitis HLS
- FPGA / Zynq SoC

---

## Project Status

Current status:

- CNN model trained successfully
- Binary DR classification implemented
- Evaluation metrics generated
- Confusion matrix generated
- Grad-CAM explainability implemented
- HLS convolution accelerator implemented
- C++ simulation of accelerator verified

Further work will focus on Vitis HLS synthesis and FPGA integration.
