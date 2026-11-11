import os  # imports os module for interacting with the operating system (e.g., file and directory operations)
import pickle  # imports pickle module for serializing and deserializing Python objects (e.g., saving and loading training history)
import time  # imports time module for working with time-related functions (e.g., generating timestamps for result directories)

import matplotlib.pyplot as plt  # imports matplotlib for plotting graphs and visualizations
import numpy as np  # imports numpy for numerical operations and array manipulations
import tensorflow as tf  # imports TensorFlow for building and training deep learning models
from sklearn.utils.class_weight import compute_class_weight # imports function from scikit learn to compute class weights for handling class imbalance
from tensorflow.keras import layers # imports layers module from Keras for building neural network layers
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small # imports pre-trained MobileNetV3 models (Large and Small variants) for transfer learning
from tensorflow.keras.optimizers import Adam  # imports Adam optimizer for training the model

# ==========================================
# 0. CONFIGURATION & REPRODUCIBILITY
# ==========================================
seed = 96 # used were 31, 42, 96
use_se = False # If True, adds a Squeeze-and-Excitation block after the base model
model_size = "small"  # Options: "small" or "large" for MobileNetV3 Small or Large models

# Set random seeds for reproducibility for tf, os and set deterministic operations for full reproducibility
tf.keras.utils.set_random_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
tf.config.experimental.enable_op_determinism()

# paths and hyperparameters
train_directory = "./handpicked/sugar/train" # path to training data
val_directory = "./handpicked/sugar/test" # path to validation data
results_directory = f"./results/MobileNetV3{model_size}_{seed}_SE-{use_se}_{time.strftime('%Y-%m-%d_%H-%M-%S')}/" # path to save results
batch_size = 32 # batch size for training and validation
epochs = 25 # number of max epochs to train
image_size = (224, 224) # image size for MobileNetV3 input
base_train = False # freezing the base model for transfer learning, if True, the base model will be trainable

# Check if results path exists and if not create it
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# ==========================================
# 1. LOAD DATASETS
# ==========================================
print("Loading training and validation datasets...")

# Loads training dataset from directory, shuffles it and resizes images to the specified image size
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

# Loads validation dataset from directory, shuffles it and resizes images to the specified image size
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_ds.class_names # get class names
num_classes = len(class_names) # count class number (10 in this case)
print(f"Found {num_classes} classes: {class_names}")

# ==========================================
# 2. CALCULATE CLASS WEIGHTS & PIPELINE MAPPING
# ==========================================
print("\nAnalyzing training label distribution...")
train_labels = [] # list to store all training labels
# Loop through the training dataset and extract labels to compute class weights
for images, labels in train_ds:
    train_labels.extend(labels.numpy())
train_labels = np.array(train_labels)

# Compute class weights to handle class imbalance in the training dataset
unique_classes = np.unique(train_labels)
computed_weights = compute_class_weight(
    class_weight='balanced', # use balanced class weights to handle class imbalance
    classes=unique_classes, # the unique classes in the training dataset
    y=train_labels # the training labels to compute class weights for
)
class_weight_dict = dict(zip(unique_classes, computed_weights)) # create a dictionary mapping class indices to their corresponding weights

print("\n==========================================")
print(" CALCULATED CLASS WEIGHTS")
print("==========================================")

# loop through the class names and print the corresponding class weights
for idx, name in enumerate(class_names):
    weight = class_weight_dict.get(idx, 1.0)
    print(f"  - Class {idx} ({name}): weight = {weight:.2f}")
print("==========================================\n")

# Convert class weights dictionary to a constant tensor for dataset mapping
class_weights_tensor = tf.constant([class_weight_dict[i] for i in range(num_classes)], dtype=tf.float32)

# Map datasets to yield (images, one_hot_labels, sample_weights)
def prepare_train_data(images, labels):
    one_hot_labels = tf.one_hot(labels, depth=num_classes)
    sample_weights = tf.gather(class_weights_tensor, labels)
    return images, one_hot_labels, sample_weights

# same for validation data but without sample weights since we don't need to weight validation samples
def prepare_val_data(images, labels):
    one_hot_labels = tf.one_hot(labels, depth=num_classes)
    return images, one_hot_labels

# apply to both datasets
train_ds = train_ds.map(prepare_train_data, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(prepare_val_data, num_parallel_calls=tf.data.AUTOTUNE)

# ==========================================
# 3. BUILD THE MODEL (Transfer Learning)
# ==========================================
print("Building MobileNetV3Small model...")

# SE block implementation
def SqueezeExcitationBlock(inputs, reduction_ratio=4):
    # Squeeze: Global spatial information into a channel descriptor
    # (keeps dimensions for proper broadcasting multiplication)
    channels = inputs.shape[-1] # get the number of channels in the input tensor
    se = layers.GlobalAveragePooling2D(keepdims=True)(inputs) # squeeze operation to get channel-wise statistics

    # Excitation: Learn channel-wise weights via bottleneck dense layers
    reduced_filters = max(1, channels // reduction_ratio) # reduce the number of filters by the reduction ratio to create a bottleneck

    dense1 = layers.Dense(reduced_filters, activation='relu', use_bias=False) # first dense layer to learn channel-wise dependencies
    dense2 = layers.Dense(channels, activation='sigmoid', use_bias=False) # second dense layer to learn channel-wise dependencies and output weights between 0 and 1

    se = dense1(se) # pass the squeezed tensor through the first dense layer
    se = dense2(se) # pass the output of the first dense layer through the second dense layer to get channel-wise weights

    # Scale original feature maps by channel-wise attention weights
    return layers.Multiply()([inputs, se]) # multiply the original input tensor with the learned channel-wise weights to scale the feature maps


# Build the model based on the specified size and whether to include SE block
if model_size == "large":
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
else:
    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

# Freeze the base model for transfer learning if base_train is False
base_model.trainable = base_train

# Define the input layer and pass it through the base model, optionally adding the SE block, followed by global average pooling and a dense output layer with softmax activation for multi-class classification
inputs = tf.keras.Input(shape=(224, 224, 3)) # define the input layer with the specified image size
x = base_model(inputs, training=base_train) # pass the inputs through the base model, setting training to base_train to control whether the base model is trainable
if use_se: # if the SE block is to be used, pass the output of the base model through the SE block
    x = SqueezeExcitationBlock(x) # pass the output of the base model through the SE block to apply channel-wise attention
x = layers.GlobalAveragePooling2D()(x) # apply global average pooling to reduce the spatial dimensions of the feature maps to a single value per channel
outputs = layers.Dense(num_classes, activation="softmax")(x) # define the output layer with the number of classes and softmax activation for multi-class classification

model = tf.keras.Model(inputs, outputs) # create the final model by specifying the inputs and outputs

# Initialize metrics (Using macro averaging for multi-class F1-score)
f1 = tf.keras.metrics.F1Score(name="f1_score", average="macro")
recall = tf.keras.metrics.Recall(name="recall")
precision = tf.keras.metrics.Precision(name="precision")

# Compile using a safer learning rate (1e-4) for fine-tuning large base models
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy", f1, recall, precision]
)

model.summary() # print the model summary to show the architecture, number of parameters, and layer details

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================

# Best model saving callback to save the model with the lowest validation loss during training
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(results_directory, "best_model.h5"),
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

# Early stopping callback to halt training if validation loss does not improve for a specified number of epochs (patience)
checkpoint_es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    mode='auto',
)

print("Starting model training with balanced sample weights...")

# train the model using the training dataset, validation dataset, specified number of epochs, and the defined callbacks for checkpointing and early stopping
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, checkpoint_es]
)

# ==========================================
# 5. PLOT & SAVE RESULTS
# ==========================================

history_path = os.path.join(results_directory, "history.pkl") # create the path to save the training history as a pickle file
with open(history_path, "wb") as f: # open the file in write-binary mode to save the training history
    pickle.dump(history.history, f) # save the training history to a pickle file for later analysis or visualization
print(f"Training history successfully saved to: {history_path}")

acc = history.history['accuracy'] # get the training accuracy values from the training history
val_acc = history.history['val_accuracy'] # get the validation accuracy values from the training history
loss = history.history['loss'] # get the training loss values from the training history
val_loss = history.history['val_loss'] # get the validation loss values from the training history
epochs_range = range(len(acc)) # create a range of epochs for plotting purposes

plt.figure(figsize=(12, 5)) # create a new figure for plotting the training and validation accuracy and loss with a specified size

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(os.path.join(results_directory, "training_plot.png"))
plt.show()

print(f"Training complete! Results and plots saved in: {results_directory}")