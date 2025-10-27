import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_reduced = df_reduced.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# separate target and feature
y = df_reduced['is_signal']
X = df_reduced.drop(columns=['type', 'is_signal', 'Unnamed0'])
print(X.head())

# split into train and test features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]
model = Sequential()

# input layer and first hidden layer
model.add(Dense(64, activation='relu', input_shape=(n_features,)))
model.add(Dropout(0.3))

# second hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(1, activation='sigmoid'))

print("\nModel architecture summary:")
model.summary()

# compile model
model.compile(optimizer='adam',  loss='binary_crossentropy',  metrics=['accuracy'])

# training
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    verbose=1 
)

# viewing training history
print("viewing training history")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt

# plotting training history
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("DNN Model Training History (Loss & Accuracy)")
plt.xlabel("Epoch")
plt.show()