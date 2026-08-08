from sklearn.preprocessing import StandardScaler
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

scaled = pd.DataFrame(StandardScaler().fit_transform(df), 
                      index=df.index, columns=df.columns)
sns.heatmap(scaled.corr(), cmap='coolwarm', center=0)