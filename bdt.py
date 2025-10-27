import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_reduced = df_reduced.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

y = df_reduced['is_signal']
X = df_reduced.drop(columns=['type', 'is_signal', 'Unnamed0'])

# Splitting into train and test sets
# 30% data kept for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)} events")
print(f"Testing set size:  {len(X_test)} events")

# LightGMB model
lgbm_classifier = lgb.LGBMClassifier(objective='binary', n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)

# Training
lgbm_classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=True)])

# plotting classifier output distribution
print("plotting classifier output distribution")
import matplotlib
import matplotlib.pyplot as plt
