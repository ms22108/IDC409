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

from sklearn.metrics import confusion_matrix
y_pred_binary = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Background', 'Signal'],
            yticklabels=['Background', 'Signal'])
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

print("applying cuts")
# Applying a cut on the BDT score
# keep 90% of signal events
signal_efficiency_target = 0.9
bdt_cut_value = np.quantile(signal_scores, 1 - signal_efficiency_target)
print(f'To achieve {signal_efficiency_target*100:.0f}% signal efficiency, the cut on BDT score is: {bdt_cut_value:.3f}')

# Applying cut to test set
test_df_with_predictions = X_test.copy()
test_df_with_predictions['is_signal'] = y_test
test_df_with_predictions['bdt_score'] = y_pred_proba
events_passing_cut = test_df_with_predictions[test_df_with_predictions['bdt_score'] > bdt_cut_value]

# Calculate background rejection
n_background_before = len(test_df_with_predictions[test_df_with_predictions['is_signal'] == 0])
n_background_after = len(events_passing_cut[events_passing_cut['is_signal'] == 0])
background_rejection = 1 - (n_background_after / n_background_before)
print(f"Background rejection at this cut: {background_rejection * 100:.2f}%")

# Plotting something after cut
def var_suppressed(feature):
    if feature in X_test.columns:
        figure, ax = plt.subplots(figsize=(6, 4))
        # Plot background before cut
        sns.histplot(test_df_with_predictions[test_df_with_predictions['is_signal'] == 0][feature],
                     bins=100, label='Background (Before Cut)', color='gray', alpha=0.5)

        
        # Plot signal for reference
        sns.histplot(test_df_with_predictions[test_df_with_predictions['is_signal'] == 1][feature],
                     bins=100, label='Signal (Reference)', color='blue', alpha=0.5, element='step', linewidth=2)
        sns.histplot(events_passing_cut[events_passing_cut['is_signal'] == 1][feature],
                     bins=100, label=f'Background (After Cut)', color='black', alpha=0.5)

