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