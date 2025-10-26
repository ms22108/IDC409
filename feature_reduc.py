import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("data_hep.csv")
df['is_signal'] = df['type'].apply(lambda x: 1 if x == 0 or x == 1 else 0)

feature_columns = df.select_dtypes(include=np.number).drop(columns=['type', 'is_signal', 'Unnamed: 0'])
corr_matrix = feature_columns.corr().abs()

sns.set(font_scale=0.9)

# plot for correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='magma', annot=False)

# avoid dropping both columns by considering upper triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# drop highly correlated features
correlation_threshold = 0.9
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
print(f"Number of features before reduction: {df.shape[1] - 3}")
print(f"Number of features to drop: {len(to_drop)}")
print(f"Features to drop: {to_drop}")
df_reduced = df.drop(columns=to_drop)
print(f"\nNumber of features after reduction: {df_reduced.shape[1] - 3}")

# plot new correlation matrix with reduced features
new_corr_matrix = df_reduced.drop(columns=['type', 'is_signal', 'Unnamed: 0']).corr().abs()
plt.figure(figsize=(10, 10))
sns.set(font_scale=0.8)
sns.heatmap(new_corr_matrix, cmap='magma', annot=False)
plt.title("New correlation matrix")

# rank reduced features
from sklearn.ensemble import RandomForestClassifier
y = df['is_signal']

#feature_columns
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(feature_columns, y)

