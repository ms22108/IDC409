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
