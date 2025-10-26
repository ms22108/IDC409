import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("data_hep.csv")

# visualising distribution of event types
print(df['type'].value_counts())
plt.figure(figsize=(4, 2))
sns.countplot(x='type', data=df)
plt.title('Distribution of Event Types')
plt.show()

# we want 0 and 1 types as signal and rest as background
df['is_signal'] = df['type'].apply(lambda x: 1 if x == 0 or x == 1 else 0) # signal is 1, rest all are background
print(df['is_signal'].value_counts())
plt.figure(figsize=(2, 2))
sns.countplot(x='is_signal', data=df)
plt.title("Binary classification: signal:1")
plt.show()