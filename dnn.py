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