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

# Evaluating the model
print("model evaluation...")
y_pred_proba = lgbm_classifier.predict_proba(X_test)[:, 1]

# Combine test results into a single DataFrame for easier plotting
results_df = pd.DataFrame({'true_label': y_test, 'bdt_score': y_pred_proba})
signal_scores = results_df[results_df['true_label'] == 1]['bdt_score']
background_scores = results_df[results_df['true_label'] == 0]['bdt_score']

# Checking for overtraining - this is the case if train curves look sharper
y_train_prob = lgbm_classifier.predict_proba(X_train)[:, 1]
train_df = pd.DataFrame({'true_label': y_train, 'bdt_score': y_train_prob})
train_signal_scores = train_df[train_df['true_label'] == 1]['bdt_score']
train_background_scores = train_df[train_df['true_label'] == 0]['bdt_score']


plt.figure(figsize=(6, 4))


plt.hist(background_scores, bins=50, range=(0, 1), label='Background',
         histtype='step', density=True, color='red', linewidth=1)
plt.hist(signal_scores, bins=50, range=(0, 1), label='Signal',
         histtype='step', density=True, color='blue', linewidth=1)

plt.hist(train_background_scores, bins=50, range=(0, 1), label='Background train',
         histtype='step', density=True, color='black', linewidth=2)
plt.hist(train_signal_scores, bins=50, range=(0, 1), label='Signal train',
         histtype='step', density=True, color='green', linewidth=2)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Classifier Output Distribution', fontsize=16)
plt.xlabel('BDT Score (Continuum Probability)', fontsize=12)
plt.ylabel('Normalized Frequency', fontsize=12)
#plt.yscale('log')
plt.legend(facecolor='white', fontsize=12)
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()

# plotting roc
print("plotting roc, confusion matrix")
fpr_bdt, tpr_bdt, thresholds = roc_curve(y_test, y_pred_proba)
auc_score_bdt = roc_auc_score(y_test, y_pred_proba)

figure, ax = plt.subplots(figsize=(5, 5))
plt.plot(fpr_bdt, tpr_bdt, label=f'BDT (AUC = {auc_score_bdt:.4f})', color='darkorange', lw=2)
plt.xlabel('Signal Efficiency (True Positive Rate)', fontsize=12)
plt.ylabel('Background Rejection (1 - False Positive Rate)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower right', facecolor='white')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
ax.set_facecolor('white')
plt.show()