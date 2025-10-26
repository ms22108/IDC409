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

