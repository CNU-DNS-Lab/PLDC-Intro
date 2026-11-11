# Plant Leaf Disease Classification (PLDC) with CNNs

This repository serves as a comprehensive resource, reference guide, and starter toolkit for **Plant Leaf Disease Classification (PLDC)** using Convolutional Neural Networks (CNNs).

---

## 📌 1. Introduction & Background

Plant diseases are responsible for severe annual losses in crop yields, directly impacting global food security. Traditional field inspection by workers is labor-intensive, time-consuming, and prone to diagnostic error. Computer vision (CV) and deep learning (DL) models, specifically **Convolutional Neural Networks (CNN)** provide automated, accurate, and lightweight solutions to assist in early disease detection and prevention.

---

## 📊 2. Dataset Fundamentals & Guidelines

Data selection and curation are critical factors in building reliable PLDC models.

### Dataset Types

* **Lab Data:** Captured in sterile, controlled environments (e.g., individual leaves on uniform backgrounds like PlantVillage). Models reach very high accuracy on lab data but often generalize poorly in real field conditions.
* **Field Data:** Captured directly in the field under real agricultural noise (soil, varying lighting, adjacent foliage, human hands). Results in robust models capable of practical deployment.
* **Hybrid Data:** Combines both lab and field conditions, providing optimal generalization across varying environments.

### Best Practices & Practical Implementation

* **Sample Size:** Ideally, if possilbe, try to aim for at least 1,000 images per class for strong model generalization (300 per class minimum).
* **Resolution:** Standard input resolution of $224 \times 224 \times 3$ is widely adopted, enabling seamless integration with pre-trained vision backbones train on ImageNet.
* **Class Imbalance:** Apply class-weighting during loss calculation, downsampling, or data augmentation (flips, rotations, color shifts) to prevent majority-class bias. [Augmentation](https://github.com/CNU-DNS-Lab/PotatoAugmentation) can be found here.
* **Bias Avoidance:** Ensure image acquisition conditions (e.g., background color, lighting, hands holding leaves) are uniformly distributed across classes to prevent the network from learning background shortcuts.

#### Practical Snippet: Data Loading

```python
import numpy as np
import tensorflow as tf

image_size = (224,224)
batch_size = 32
data_directory = "./data/dataset/train"
test_directory = "./data/dataset/test"
```

- Load training and validation data from same directory with split.

```python
# 1. Load datasets from directory structure
# Loads training portion of dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    label_mode='categorical', 
    validation_split=0.2,
    subset="training",
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

# Loads validation portion of dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    label_mode='categorical',
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

```

- Load train, validation, and test data. Test data is in separate directory.

```python
# Loads training portion of dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    label_mode='categorical', 
    validation_split=0.2,
    subset="training",
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

# Loads validation portion of dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    validation_split=0.2,
    subset="validation",
    label_mode='categorical',
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)

# Loads training portion of dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    label_mode='categorical', 
    seed=1337,
    color_mode="rgb",
    image_size=image_size,
    batch_size=batch_size,
)
```

---

## 🏗️ 3. CNN Architectures & Advanced Modules

### Established Model Families in PLDC

* **ConvNeXt:** Modernized pure-CNN architecture incorporating design choices from Vision Transformers (ViT) such as large kernels, depthwise convolutions, and GELU activations.
* **DenseNet:** Connects each layer to every subsequent layer to maximize feature reuse, mitigate vanishing gradients, and reduce overall parameter count.
* **EfficientNetV2:** Utilizes MBConv and Fused-MBConv structures optimized via neural architecture search for parameter efficiency and fast training.
* **MobileNetV3:** Highly optimized lightweight model designed for deployment on edge and mobile hardware using depthwise separable convolutions and Squeeze-and-Excitation blocks.

* **etc.**

### Key Architectural Modules

* **Depthwise Separable Convolutions (DSConv):** Splits standard convolution into depthwise spatial filtering and pointwise channel mixing, reducing computational FLOPs significantly.
* **Residual Connections (RC):** Adds input feature maps directly to layer outputs, stabilizing gradient flow across deeper networks.
* **Attention Mechanisms:**
  * **Squeeze-and-Excitation (SE):** Dynamically recalibrates channel-wise feature map importances.
  * **CBAM (Convolutional Block Attention Module):** Sequentially applies spatial and channel attention to focus on critical regions of the image. Channel and Spatial attention can also be used indpendantly.
* **etc.**

#### Practical Snippet: Basic CNN examples using TensorFlow


##### Simple CNN example in sequential API

```python
import tensorflow as tf

model_sequential = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model_sequential.summary()
```

##### Simple CNN example in funtional API


```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model_functional = tf.keras.Model(inputs=inputs, outputs=outputs)

model_functional.summary()
```

##### Simple CNN based on existing model

```python
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

# Build Transfer Learning Model with Backbone + SE Layer
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = True

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
```

---

## ⚙️ 4. Training Strategies & Best Practices

1. **Transfer Learning (TL):** Initialize models with pre-trained weights to speed up convergence and train effectively on smaller datasets (often ImageNet is used).
2. **Domain-Related (DR) Pre-training:** Pre-train backbones on large-scale domain-specific datasets (e.g., PLDC-80) before fine-tuning on targeted tasks to bridge the domain gap.
3. **Reproducibility:** Fix random seeds across Python, NumPy, and TensorFlow environments (`tf.config.experimental.enable_op_determinism()` and seed usage).
4. **Validation:** Implement Early Stopping and Model Checkpointing to restore the best weights and prevent overfitting. Evaluate using Precision, Recall, F1-Score, and AUC alongside standard Accuracy.

#### Practical Snippet: Model Compilation, Callbacks & Training Loop

```python
from tensorflow.keras.optimizers import Adam


# Compile model with Adam optimizer and comprehensive evaluation metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy", tf.keras.metrics.F1Score(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Define early stopping and checkpoint callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# Train with balanced class weights
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weight_dict
)
```

---

#### Practical Snippet: Determinism & Randomness Management


#### Implementation Example (TensorFlow / Keras)

```python
import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    # 1. Set standard Python & NumPy random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. Set TensorFlow environment random seed
    tf.keras.utils.set_random_seed(seed)
    
    # 3. Force deterministic behavior for Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 4. Enforce deterministic GPU kernel operations in TensorFlow
    tf.config.experimental.enable_op_determinism()

# Call prior to dataset loading or model initialization
set_seed(42)
```

---

## 💻 6. Example Code

While this repository emphasizes general PLDC concepts, a complete end-to-end reference script incorporating all steps above is provided in `simple.py`.

### Repository Structure
To run the experiments set up your directory as follows:
- Dataset can be found [here](doi.org/10.17632/9twjtv92vk.1). Changes made to the dataset can be found in paper.


```text
PLDC-Intro/
├── simple.py              # Reference implementation (MobileNetV3 + SE Block + Class Weights)
├── README.md              # Main documentation
└── data/            # Example dataset directory structure
    └── sugar/
        ├── train/         # Training images (grouped by class folders)
        └── test/          # Testing images (grouped by class folders)
```

### Quickstart

1. **Install dependencies:**
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
   ```

2. **Run the example training pipeline:**
   ```bash
   python simple.py
   ```

---

## 📖 Citation

If you reference these guidelines or use this code in your research, please cite our paper:

```bibtex
@inproceedings{richter2026pldcintro,
  title={CNN for Plant Leaf Disease Classification: An Introduction},
  author={Richter, David J. and Kim, Kyungbaek},
  year={2026}
}
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.