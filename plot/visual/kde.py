import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
matrix = np.loadtxt("cap.txt")
mean = np.mean(matrix)
std = np.std(matrix)
z_score_matrix = (matrix - mean) / std
sns.kdeplot(z_score_matrix, bw_adjust=0.5)
values = [6, 4, 4, 6, 6, 7, 5, 4, 6, 6]
for value in z_score_matrix:
  plt.axvline(x=value, color='red', linestyle='--', alpha=0.45)
unique = np.unique(values).flatten()
for value in unique:
  plt.axhline(y=value, color='red', linestyle='--', alpha=0.45)
x = np.linspace(0, 0.9, 10)
dist = stats.norm
params = dist.fit(z_score_matrix)
pdf = dist.pdf(x, *params)
plt.plot(x, pdf, color='blue', linestyle='-', linewidth=1)
plt.xlabel('X')
plt.ylabel('KDE')
plt.title('MLE')
plt.xlim(0, 0.9)
plt.ylim(0, 0.9)
plt.show()
