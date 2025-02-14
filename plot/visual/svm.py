import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
def calculate_lof_scores(data):
  lof = LocalOutlierFactor(n_neighbors=5, algorithm='auto', metric='euclidean')
  lof.fit(data)
  lof_scores = lof.negative_outlier_factor_
  return lof_scores
def phi(distance):
  return np.exp(-distance**2)
#Z = [6, 5, 6, 8, 5, 4, 6, 4, 4, 6] calf3
Z = [6, 4, 4, 6, 6, 7, 5, 4, 6, 6]
def adjust_mean(Z, mean):
  current_mean = np.mean(Z)
  adjustment_factor = mean - current_mean
  for i in range(len(Z)):
    Z[i] += adjustment_factor
adjust_mean(Z, 6)
X = [item for item in Z]
print(X[:len(X)//2])
print(X[len(X)//2:])
with open("cap.txt", "w") as f:
  for element in X:
    fm_element = "{:.2f}".format(element)
    f.write(str(fm_element) + " ")
matrix = np.loadtxt("cap.txt")
mean = np.mean(matrix)
std = np.std(matrix)
z_score_matrix = (matrix - mean) / std
X = np.reshape(z_score_matrix, (len(z_score_matrix)))
X = [round(item, 1) for item in X]
print(X[:len(X)//2])
print(X[len(X)//2:])
y = np.array([0 for i in range(len(z_score_matrix))])
X = np.reshape(X, (5, 2))
y = np.array([0, 0, 1, 1, 1])
clf = svm.SVC(kernel='rbf', gamma='scale', probability=True)
clf.fit(X, y)
h = 0.008 #0.001
x_min, x_max = X[:, 0].min() - h, X[:, 0].max() + h
y_min, y_max = X[:, 1].min() - h, X[:, 1].max() + h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = list(np.c_[xx.ravel(), yy.ravel()])
probas = clf.predict_proba(grid_points)
probas = probas[:, 1].reshape(xx.shape)
decision_boundary_values = [0.4, 0.5, 0.6, 0.7, 0.9]
decision_boundaries = []
for value in decision_boundary_values:
  decision_boundary = np.where(probas >= value, 1, 0)
  decision_boundaries.append(decision_boundary)
fig, ax = plt.subplots()
ax.contourf(xx, yy, probas, levels=[-1, 0, 1], colors=['lightblue', 'lightcoral'])
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
for decision_boundary in decision_boundaries:
  ax.contour(xx, yy, decision_boundary, levels=[0.5], colors='black', alpha=0.9)
lof_scores = calculate_lof_scores(X)
threshold = 6
outliers = np.where(lof_scores < threshold)[0]
ax.scatter(X[outliers, 0], X[outliers, 1], c='yellow', marker='x', s=100)
ax.scatter(X[:, 0][:3], X[:, 1][:3], c='red', marker='o', s=100)
ax.scatter(X[:, 0][3:5], X[:, 1][3:5], c='green', marker='s', s=100)
ax.scatter(X[:, 0][5:], X[:, 1][5:], c='blue', marker='^', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision with RBF Kernel and LOF')
plt.show()
