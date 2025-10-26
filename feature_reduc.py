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


