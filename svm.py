import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_reduced = df_reduced.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# separate target and feature
y = df_reduced['is_signal']
X = df_reduced.drop(columns=['type', 'is_signal', 'Unnamed0'])

# split into train and test features
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# we are going to plot roc 
print("plotting ROC, confusion matrix...")
from sklearn.metrics import (classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score)
import matplotlib.pyplot as plt
import numpy as np

y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# precision, recall, f1 score
print(classification_report(y_test, y_pred, target_names=['Background (0)', 'Signal (1)']))