from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

y = df_reduced['is_signal']
X = df_reduced.drop(columns=['type', 'is_signal', 'Unnamed: 0'])

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, # Use the scaled data
    test_size=0.3,
    random_state=42,
    stratify=y)

# log reg classifier
log_reg_classifier = LogisticRegression(
    solver='liblinear',
    C=0.1,
    random_state=42,
    max_iter=1000 # increase iterations for convergence with many features
)
log_reg_classifier.fit(X_train, y_train)

y_pred_proba = log_reg_classifier.predict_proba(X_test)[:, 1]

# combine results into a dataframe for plotting
results_df = pd.DataFrame({'true_label': y_test, 'logreg_score': y_pred_proba})
signal_scores = results_df[results_df['true_label'] == 1]['logreg_score']
background_scores = results_df[results_df['true_label'] == 0]['logreg_score']

# classifier output
figure, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('white')
plt.hist(background_scores, bins=50, range=(0, 1), label='Background (Continuum)',
         histtype='step', density=True, color='red', linewidth=2)
plt.hist(signal_scores, bins=50, range=(0, 1), label='Signal',
         histtype='step', density=True, color='blue', linewidth=2)
plt.title('Logistic Regression Classifier Output', fontsize=16)
plt.xlabel('Model Score (Probability)', fontsize=12)
plt.ylabel('Normalized Frequency', fontsize=12)
plt.legend(facecolor='white')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# plotting roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba) # area

figure, ax = plt.subplots(figsize=(6, 5))
ax.set_facecolor('white')
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.4f})', color='purple', lw=2)
plt.xlabel('Signal Efficiency (True Positive Rate)', fontsize=12)
plt.ylabel('Background Rejection (False Positive Rate)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc='lower left', facecolor='white')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()

y_pred_binary = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)

figure, ax = plt.subplots(figsize=(5, 5))
ax.set_facecolor('white')
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Background', 'Signal'],
            yticklabels=['Background', 'Signal'])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

# Applying cuts on the distribution
print("Applying cuts on the distribution")
signal_efficiency_target = 0.90
cut_value = np.quantile(signal_scores, 1 - signal_efficiency_target)
print(f"To achieve {signal_efficiency_target*100:.0f}% signal efficiency, the cut is: {cut_value:.3f}")

n_background_before = len(background_scores)
n_background_after = (background_scores > cut_value).sum()
background_rejection = 1 - (n_background_after / n_background_before)
print(f"Background rejection at this cut: {background_rejection * 100:.2f}%")

def var_suppressed(feature):
    if feature in X.columns:
        # Reconstruct a DataFrame with the original (unscaled) test data and predictions
        test_df_with_predictions = X.loc[y_test.index].copy()
        test_df_with_predictions['is_signal'] = y_test
        test_df_with_predictions['logreg_score'] = y_pred_proba

	# Separate signal and background for plotting
        signal_before_cut = test_df_with_predictions[test_df_with_predictions['is_signal'] == 1]
        background_before_cut = test_df_with_predictions[test_df_with_predictions['is_signal'] == 0]
    
        # Apply the cut
        events_passing_cut = test_df_with_predictions[test_df_with_predictions['logreg_score'] > cut_value]
        signal_after_cut = events_passing_cut[events_passing_cut['is_signal'] == 1]
        background_after_cut = events_passing_cut[events_passing_cut['is_signal'] == 0]

    
       
        plt.figure(figsize=(10, 6))
        sns.histplot(data=background_before_cut, x=feature, bins=100, 
                     label='Background before cut', color='grey', alpha=0.7)
        sns.histplot(data=signal_before_cut, x=feature, bins=100, 
                     label='Signal before cut', color='blue', alpha=0.7, element='step', linewidth=2)
