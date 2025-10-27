import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_reduced = df_reduced.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

