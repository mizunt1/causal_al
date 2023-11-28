import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('erm_2_partial.csv', index_col=0)
#df['proportion'] = np.where(df['proportion'] <=1, df['proportion'], df['proportion']/1000)
plt.plot(df['proportion'], df['worst acc'], label='erm')
df = pd.read_csv('irm_2_partial.csv', index_col=0)
#df['proportion'] = np.where(df['proportion'] <=1, df['proportion'], df['proportion']/1000)
plt.plot(df['proportion'], df['worst acc'], label='irm')
plt.ylabel("worst group accuracy") 
plt.xlabel("amount of environment one") 
plt.title("effect of increasing env1 proportion on worst group accuracy") 
  
# Adding legend, which helps us recognize the curve according to it's color 
plt.legend() 
plt.show()
