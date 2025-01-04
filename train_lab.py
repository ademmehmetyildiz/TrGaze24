# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 08:20:42 2024

@author: PC
"""
import math
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# Extract the numerical features (11 additional inputs)


# Extract the target variables
#y = data[['g_pitch_ethx', 'g_yaw_ethx']].values
#y[:, 0] = y[:, 0] * 34.54   # g_pitch_ethx için
#y[:, 1] = y[:, 1] * 19.43  # g_yaw_ethx için

"""
yp = data[['g_pitch_ethx']].values
yy = data[['g_yaw_ethx']].values

ypha = np.radians(yy)
ytheta = np.radians(yp)


gx = np.cos(ypha) * np.cos(ytheta)
gy = np.cos(ypha) * np.sin(ytheta)
gz = np.sin(ypha)
yson = np.sqrt(gx**2 + gy**2 + gz**2)
"""


"""

ythetaleft = data[['theta_l']].values
ythetaright = data[['theta_r']].values

yphaleft = data[['pha_l']].values
ypharight = data[['pha_r']].values



thetaleft = np.radians(ythetaleft)
thetaright = np.radians(ythetaright)

phaleft= np.radians(yphaleft)
pharight= np.radians(ypharight)


gxl = np.cos(phaleft) * np.cos(thetaleft)
gyl = np.cos(phaleft) * np.sin(thetaleft)
gzl = np.sin(phaleft)

gxr = np.cos(pharight) * np.cos(thetaright)
gyr = np.cos(pharight) * np.sin(thetaright)
gzr = np.sin(pharight)


yson = (np.sqrt(gxl**2 + gyl**2 + gzl**2)/np.sqrt(gxr**2 + gyr**2 + gzr**2))
"""
#y[:, 0] = y[:, 0] * 1920  # g_pitch_ethx için
#y[:, 1] = y[:, 1] * 1080  # g_yaw_ethx için



#y = scaler.fit_transform(y)

#X_numeric = data.drop(columns=['left_url', 'right_url', 'x', 'y']).values
#X_numeric = data[['pitch','yaw','roll']].values
"""
X_numeric = data[['pitch','yaw','roll',
                  'theta_l','theta_r',
                  'pha_l','pha_r',
                  'face_bbox_0',
                  'face_bbox_1',
                  'face_bbox_2',
                  'face_bbox_3']].values
"""

data = pd.read_csv('data5.csv')
# Define image size and channels
image_size = (64, 64)
channels = 1  # Grayscale

# Function to load and preprocess images
def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, image_size)  # Resize to match the input shape
    img = img / 255.0  # Normalize pixel values
    return img

# Load and preprocess the images
X_left = np.array([load_image(f"{img}") for img in data['left_url']])
X_right = np.array([load_image(f"{img}") for img in data['right_url']])

# Stack left and right images along the channel dimension
X_images = np.stack([X_left, X_right], axis=-1)

y = data[['g_pitch_ethx', 'g_yaw_ethx']].values
#y[:, 0] = y[:, 0] * 34.5   # g_pitch_ethx için
#y[:, 1] = y[:, 1] * 19.4  # g_yaw_ethx için
X_numeric = data[['pitch','yaw','roll',
                  'theta_l','theta_r',
                  'pha_l','pha_r',
                  'face_bbox_0',
                  'face_bbox_1',
                  'face_bbox_2',
                  'face_bbox_3']].values
# Split the dataset into training and test sets
X_images_train, X_images_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_images, X_numeric, y, test_size=0.2, random_state=42)

# Build the CNN for image data
image_input = Input(shape=(64, 64, 2))
cnn = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = Conv2D(64, kernel_size=(3, 3), activation='relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2))(cnn)
cnn = Flatten()(cnn)

# Build the fully connected layers for numerical data
numeric_input = Input(shape=(11,))
dense = Dense(64, activation='relu')(numeric_input)
dense = Dense(32, activation='relu')(dense)

# Combine the CNN and numerical input
combined = Concatenate()([cnn, dense])
output = Dense(2)(combined)  # Output for x and y coordinates

# Build and compile the model
model = Model(inputs=[image_input, numeric_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit([X_images_train, X_numeric_train], y_train, validation_data=([X_images_test, X_numeric_test], y_test), epochs=20, batch_size=64)

# Plot the loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Mean Absolute Error (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Save the model
model.save('data5_son_makale_nokta_açı.h5')
