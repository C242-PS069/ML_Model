# Waste Classification Model

Deep learning model for classifying waste images into recyclable material categories using transfer learning.
The model identifies different types of waste from images to support automated waste sorting systems.

This project was developed as part of the **Bangkit Academy Machine Learning pathway**, a program led by Google, GoTo, and Traveloka.

---

# Project Overview

Waste management is a critical challenge in many regions. Manual sorting is inefficient and prone to error.
This project explores how computer vision can assist in automatically classifying waste materials from images.

The model performs **multi-class image classification** to detect common recyclable materials.

---

# Waste Categories

The model classifies images into the following categories:

* cardboard
* glass
* metal
* paper
* plastic_bottle
* plastic_container
* plastic_cup

Total classes: **7**

---

# Model Architecture

The model uses **transfer learning** with a pretrained MobileNetV2 backbone.

Pipeline architecture:

```
MobileNetV2 (pretrained)
        ↓
GlobalAveragePooling2D
        ↓
Dense (128)
        ↓
Dense (7) – Softmax
```

Model specifications:

* Backbone: MobileNetV2
* Input size: **224 × 224**
* Output classes: **7**
* Total parameters: **~2.75M**
* Trainable parameters: **~165K**

The base MobileNetV2 layers are frozen while training the classification head.

---

# Dataset Structure

The dataset uses a directory-based format for image classification:

```
dataset/
├── cardboard
├── glass
├── metal
├── paper
├── plastic_bottle
├── plastic_container
└── plastic_cup
```

Each folder contains images representing its corresponding class label.

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Google Colab
* Transfer Learning with MobileNetV2

---

# Project Structure

```
ML_Model/
│
├── dataset/
│   ├── cardboard
│   ├── glass
│   ├── metal
│   ├── paper
│   ├── plastic_bottle
│   ├── plastic_container
│   └── plastic_cup
│
├── model.h5
│
├── inference.py
│
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/C242-PS069/ML_Model.git
cd ML_Model
```

Install dependencies:

```
pip install tensorflow numpy
```

---

# Load the Model

Example code to load the trained model:

```python
from tensorflow.keras.models import load_model

model = load_model("model.h5")
model.summary()
```

---

# Example Prediction

Run inference on a sample image:

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("model.h5")

img = image.load_img("sample.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

print(prediction)
```

---

# Training Approach

The model was trained using transfer learning:

1. Load pretrained MobileNetV2
2. Freeze base convolution layers
3. Add classification head
4. Train Dense layers on waste dataset
5. Optimize using categorical cross-entropy loss

---

# Potential Applications

This model can be extended for:

* smart recycling bins
* automated waste sorting systems
* environmental monitoring tools
* mobile recycling applications

---

# Future Improvements

Possible improvements include:

* expanding dataset size
* adding more waste categories
* improving model accuracy through fine-tuning
* deploying the model as a REST API
* integrating with mobile or web applications

---

# License

This project is intended for educational and research purposes.
